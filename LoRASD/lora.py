import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image

from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler


from peft import LoraConfig, get_peft_model

CONFIG = {
    "pretrained_model_name_or_path": "stable-diffusion-v1-5-lora",
    "train_data_dir": "bottle",
    "output_dir": "./output_lora_bottle",
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "max_train_steps": 1000,
    "save_steps": 500,
    "mixed_precision": "fp16",
    "seed": 42,
    "trigger_word": "broken bottle",
}

class MVTecDataset(Dataset):

    def __init__(self, root_dir, tokenizer, size=512, trigger_word="object"):
        self.size = size
        self.tokenizer = tokenizer
        self.image_paths = []
        self.prompts = []

        defect_dir = os.path.join(root_dir, "test", "broken_large")
        if os.path.exists(defect_dir):
            for f in os.listdir(defect_dir):
                if f.endswith(".png"):
                    self.image_paths.append(os.path.join(defect_dir, f))
                    self.prompts.append(f"photo of a {trigger_word}, defect, anomaly")

        good_dir = os.path.join(root_dir, "train", "good")
        good_images = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith(".png")]

        good_images = good_images[:len(self.image_paths)]
        for img_path in good_images:
            self.image_paths.append(img_path)
            self.prompts.append(f"photo of a good {trigger_word.split()[-1]}, flawless")

        print(f"Dataset Loaded: {len(self.image_paths)} images.")

        self.transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transforms(image)


        prompt = self.prompts[idx]
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids[0]
        }

# =================主函数=================
def main():
    device = "cuda"
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading models...")
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model_name_or_path"], subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["pretrained_model_name_or_path"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["pretrained_model_name_or_path"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(CONFIG["pretrained_model_name_or_path"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(CONFIG["pretrained_model_name_or_path"], subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float16 if CONFIG["mixed_precision"] == "fp16" else torch.float32
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device)

    print("Injecting LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = MVTecDataset(CONFIG["train_data_dir"], tokenizer, size=CONFIG["resolution"], trigger_word=CONFIG["trigger_word"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["train_batch_size"], shuffle=True)


    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=CONFIG["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=CONFIG["max_train_steps"]
    )


    print("Starting training...")
    global_step = 0
    unet.train()

    progress_bar = tqdm(range(CONFIG["max_train_steps"]))

    while global_step < CONFIG["max_train_steps"]:
        for step, batch in enumerate(dataloader):

            latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"] == "fp16"):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss.backward()

            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})

                if global_step % CONFIG["save_steps"] == 0:
                    save_path = os.path.join(CONFIG["output_dir"], f"checkpoint-{global_step}")
                    unet.save_pretrained(save_path)
                    print(f"\nSaved LoRA to {save_path}")

            if global_step >= CONFIG["max_train_steps"]:
                break

    final_save_path = os.path.join(CONFIG["output_dir"], "final_lora")
    unet.save_pretrained(final_save_path)
    print("Training finished!")

if __name__ == "__main__":
    main()