import cv2
import numpy as np
import os
import random
from glob import glob
from PIL import Image
from safetensors.torch import load_file

from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPVisionModelWithProjection

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from mask_gene import generate_meta_mask

import torch
import torch.nn.functional as F

class LMAController:

    def __init__(self, unet, amplification_factor=1.5):
        self.unet = unet
        self.amplification_factor = amplification_factor
        self.mask = None
        self.hooks = []
        self.active = False

    def set_mask(self, mask_tensor):
        self.mask = mask_tensor

    def enable(self):
        if self.active: return
        self.active = True
        self.hooks = []

        for i, block in enumerate(self.unet.up_blocks):
            for resnet in block.resnets:
                handle = resnet.register_forward_hook(self._hook_fn)
                self.hooks.append(handle)

        if self.unet.mid_block:
            for resnet in self.unet.mid_block.resnets:
                handle = resnet.register_forward_hook(self._hook_fn)
                self.hooks.append(handle)

    def disable(self):
        if not self.active: return
        self.active = False
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def _gaussian_blur_mask(self, mask, kernel_size=5, sigma=1.5):
        channels = mask.shape[1]
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(mask.device, dtype=mask.dtype)

        pad = (kernel_size - 1) // 2
        return F.conv2d(mask, gaussian_kernel, padding=pad, groups=channels)

    def _hook_fn(self, module, input, output):
        if self.mask is None or not self.active:
            return output

        if isinstance(output, tuple):
            h_states = output[0]
        else:
            h_states = output

        orig_dtype = h_states.dtype
        b, c, h, w = h_states.shape

        mask_down = F.interpolate(
            self.mask.to(dtype=orig_dtype),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        if h >= 16:
            mask_down = self._gaussian_blur_mask(mask_down, kernel_size=5, sigma=1.0)

        if h > 8:
            scale_map = 1.0 + (self.amplification_factor - 1.0) * mask_down

            h_states = h_states * scale_map

            h_states = torch.clamp(h_states, min=-8.0, max=8.0)

        if isinstance(output, tuple):
            return (h_states,) + output[1:]
        else:
            return h_states

class SCCA_IPAdapterAttnProcessor(torch.nn.Module):

    def __init__(self, hidden_size, cross_attention_dim, scale=1.0, num_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.to_k_ip = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = torch.nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.scca_mask = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        elif attn.norm_cross: encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.scale > 0:
            key_ip = self.to_k_ip(encoder_hidden_states)
            value_ip = self.to_v_ip(encoder_hidden_states)

            key_ip = attn.head_to_batch_dim(key_ip)
            value_ip = attn.head_to_batch_dim(value_ip)

            attn_probs_ip = attn.get_attention_scores(query, key_ip, None)
            hidden_states_ip = torch.bmm(attn_probs_ip, value_ip)
            hidden_states_ip = attn.batch_to_head_dim(hidden_states_ip)

            if self.scca_mask is not None:
                h = int(sequence_length**0.5)

                mask_down = F.interpolate(self.scca_mask.to(dtype=query.dtype), size=(h, h), mode="bilinear", align_corners=False)
                mask_flat = mask_down.view(1, -1, 1)

                hidden_states_ip = hidden_states_ip * mask_flat

                mask_ratio = mask_flat.mean()
                boost_factor = 1.0
                if mask_ratio > 1e-6 and mask_ratio < 0.1:
                    boost_factor = 1.0 + (0.1 - mask_ratio) * 30.0
                    boost_factor = min(boost_factor.item(), 4.0)

                hidden_states_ip = hidden_states_ip * boost_factor

            hidden_states = hidden_states + scale * self.scale * hidden_states_ip

        # 🛠️ 类型修复
        if hasattr(attn.to_out[0], "weight"):
            hidden_states = hidden_states.to(dtype=attn.to_out[0].weight.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection: hidden_states = hidden_states + residual
        return hidden_states


DEVICE = "cuda"
SEED = 42

# 路径配置
# 路径配置
SD_MODEL_PATH = "/stable-diffusion-v1-5"
IMAGE_ENCODER_PATH = "/IP-Adapter/models/image_encoder"
CHECKPOINT_ROOT = "/checkpoints/IP-Adaptor-Final" # IP-Adapter root
LORA_ROOT = "/checkpoints/LoRA-checkpoints"            # LoRA root (output_loras_all 或 output_lora_{obj})
SAM2_CHECKPOINT = "/checkpoints/checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"
OUTPUT_ROOT_DIR = "generated_dataset" # 总输出目录

NUM_GEN_PER_ANOMALY = 1000
M_MAX = 5
K_SHOT = 5


OBJECTS = ["bottle"]


OBJECT_ANOMALIES = {
    "bottle": ["broken_large", "broken_small", "contamination"]
}

OBJ_CONFIG = {
    "bottle":     {"strength": 0.9, "lora_scale": 0.85, "ip_range": (0.6, 0.85), "cfg": 8.0, "use_sam2": True, "lma_factor": 1.1},
 }
DEFAULT_CONFIG = {"strength": 0.9, "lora_scale": 0.8, "ip_range": (0.6, 0.8), "cfg": 8.0, "use_sam2": True, "lma_factor": 1.3}

# ==========================================
# 3. 辅助函数
# ==========================================
def init_models():
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IMAGE_ENCODER_PATH, subfolder="", torch_dtype=torch.float16
    ).to(DEVICE)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL_PATH, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False, image_encoder=image_encoder
    ).to(DEVICE)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)
    return pipe, predictor

def extract_smart_patch(real_img, mask):
    ys, xs = np.where(mask > 50)
    if len(ys) == 0: return Image.fromarray(real_img).resize((224, 224))
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h_defect, w_defect = y_max - y_min, x_max - x_min
    pad_base = 5
    pad_ratio = 0.02
    pad_y = int(max(pad_base, h_defect * pad_ratio)); pad_x = int(max(pad_base, w_defect * pad_ratio))
    H_img, W_img = real_img.shape[:2]
    y1 = max(0, y_min - pad_y); y2 = min(H_img, y_max + pad_y)
    x1 = max(0, x_min - pad_x); x2 = min(W_img, x_max + pad_x)
    patch_np = real_img[y1:y2, x1:x2]
    patch_pil = Image.fromarray(patch_np)
    if patch_pil.width < 32 or patch_pil.height < 32: return Image.fromarray(real_img).resize((224, 224))
    return patch_pil

def get_lora_prompt(obj, anomaly):
    if anomaly == "combined":
        anomaly_desc = "combined anomalies"
    else:
        anomaly_desc = anomaly.replace("_", " ")
    prompt = f"photo of a {anomaly_desc} {obj}, defect, anomaly"
    negative_prompt = "cartoon, blurry, low quality, distortion, watermarked, good condition, flawless, deformed"
    return prompt, negative_prompt


def main():
    pipe, predictor = init_models()

    lma_controller = LMAController(pipe.unet, amplification_factor=2.0)

    for obj in OBJECTS:

        cfg = OBJ_CONFIG.get(obj, DEFAULT_CONFIG)
        use_foreground = cfg["use_sam2"]

        pipe.unload_lora_weights()
        lora_dir = os.path.join(LORA_ROOT, f"output_lora_{obj}", "final_lora")
        lora_file = os.path.join(lora_dir, "adapter_model.safetensors")

        has_lora = False
        if os.path.exists(lora_file):

            try:
                state_dict = load_file(lora_file)
                new_state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
                pipe.load_lora_weights(new_state_dict, adapter_name=obj)
                pipe.fuse_lora(lora_scale=cfg["lora_scale"])
                has_lora = True
            except Exception as e:
                print(f"error")
        else:
            print(f"no file")

        adapter_folder = os.path.join(CHECKPOINT_ROOT, f"ip_adapter_{obj}", "checkpoint-3000")
        adapter_file = "ip_adapter.bin"
        if not os.path.exists(os.path.join(adapter_folder, adapter_file)):
            # 尝试寻找 checkpoint-final
            adapter_folder = os.path.join(CHECKPOINT_ROOT, f"ip_adapter_{obj}", "checkpoint-final")

        if os.path.exists(os.path.join(adapter_folder, adapter_file)):
            pipe.load_ip_adapter(adapter_folder, subfolder="", weight_name=adapter_file)

            scca_processors = {}
            for name, proc in pipe.unet.attn_processors.items():
                if hasattr(proc, "to_k_ip"):
                    hidden_dim = proc.to_k_ip.out_features
                    cross_dim = proc.to_k_ip.in_features
                    new_proc = SCCA_IPAdapterAttnProcessor(hidden_dim, cross_dim, scale=1.0)
                    new_proc.to_k_ip.load_state_dict(proc.to_k_ip.state_dict())
                    new_proc.to_v_ip.load_state_dict(proc.to_v_ip.state_dict())
                    new_proc.to(DEVICE, dtype=torch.float16)
                    scca_processors[name] = new_proc
                else:
                    scca_processors[name] = proc
            pipe.unet.set_attn_processor(scca_processors)
        else:
            print(f"no file adapter")
            continue

        anomalies = OBJECT_ANOMALIES.get(obj, [])
        for anomaly in anomalies:

            all_normal_images = glob(f"/mvtec/{obj}/train/good/*.png")
            all_anomaly_ref_images = sorted(glob(f"/mvtec/{obj}/test/{anomaly}/*.png"))
            available_ref_images = all_anomaly_ref_images[:K_SHOT]

            if not all_normal_images or not available_ref_images:
                continue

            save_dir = os.path.join(OUTPUT_ROOT_DIR, obj)
            test_dir = os.path.join(save_dir, "test", anomaly)
            gt_dir = os.path.join(save_dir, "ground_truth", anomaly)
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            for i in range(NUM_GEN_PER_ANOMALY):
                init_image_path = random.choice(all_normal_images)
                patch_image_path = random.choice(available_ref_images)

                try:
                    init_image_cv2 = cv2.imread(init_image_path)
                    init_image_cv2 = cv2.cvtColor(init_image_cv2, cv2.COLOR_BGR2RGB)
                    h_orig, w_orig = init_image_cv2.shape[:2]

                    real_img = cv2.imread(patch_image_path)
                    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                    mask_path = patch_image_path.replace(".png", "_mask.png").replace("test", "ground_truth")

                    if os.path.exists(mask_path):
                        mask_ref = cv2.imread(mask_path, 0)
                        patch_pil = extract_smart_patch(real_img, mask_ref)
                    else:
                        patch_pil = Image.fromarray(real_img)
                except Exception as e:
                    print(f"      Error: {e}")
                    continue

                foreground_mask = np.ones((h_orig, w_orig), dtype=np.uint8) * 255
                if use_foreground:
                    try:
                        predictor.set_image(init_image_cv2)
                        input_points = np.array([[0, 0], [w_orig-1, 0], [0, h_orig-1], [w_orig-1, h_orig-1], [0, h_orig//2], [w_orig-1, h_orig//2]])
                        input_labels = np.ones(len(input_points))
                        masks, scores, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
                        best_mask = masks[np.argmax(scores)].astype(bool)
                        foreground_mask = (~best_mask).astype(np.uint8) * 255
                        m = 3
                        foreground_mask[:m,:]=0; foreground_mask[-m:,:]=0; foreground_mask[:,:m]=0; foreground_mask[:,-m:]=0
                    except: pass

                final_anomaly_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                for _ in range(10):
                    raw_anomaly = generate_meta_mask(W=w_orig, H=h_orig, m_max=M_MAX)
                    intersection = cv2.bitwise_and(raw_anomaly, foreground_mask)
                    if cv2.countNonZero(intersection) > 50:
                        final_anomaly_mask = intersection
                        break

                kernel = np.ones((3, 3), np.uint8)
                final_anomaly_mask = cv2.dilate(final_anomaly_mask, kernel, iterations=1)

                mask_tensor = torch.from_numpy(final_anomaly_mask).to(DEVICE, dtype=torch.float16) / 255.0
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

                for proc in pipe.unet.attn_processors.values():
                    if isinstance(proc, SCCA_IPAdapterAttnProcessor):
                        proc.scca_mask = mask_tensor

                lma_controller.set_mask(mask_tensor)
                lma_controller.amplification_factor = cfg["lma_factor"]
                lma_controller.enable()

                ip_min, ip_max = cfg["ip_range"]
                rand_ip_scale = random.uniform(ip_min, ip_max)
                pipe.set_ip_adapter_scale(rand_ip_scale)
                prompt, negative_prompt = get_lora_prompt(obj, anomaly)

                result_pil = pipe(
                    prompt=prompt, negative_prompt=negative_prompt,
                    image=Image.fromarray(init_image_cv2), mask_image=Image.fromarray(final_anomaly_mask),
                    ip_adapter_image=patch_pil,
                    height=512, width=512,
                    strength=cfg["strength"],
                    guidance_scale=cfg["cfg"],
                    num_inference_steps=200, generator=torch.Generator(device=DEVICE).manual_seed(SEED + i)
                ).images[0]

                lma_controller.disable()

                for proc in pipe.unet.attn_processors.values():
                    if isinstance(proc, SCCA_IPAdapterAttnProcessor):
                        proc.scca_mask = None

                result_cv2 = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                result_cv2 = cv2.resize(result_cv2, (w_orig, h_orig))
                init_img_bgr = cv2.cvtColor(init_image_cv2, cv2.COLOR_RGB2BGR)
                mask_binary = (final_anomaly_mask > 127).astype(np.float32)
                mask_binary = np.expand_dims(mask_binary, axis=2)
                composite = init_img_bgr * (1 - mask_binary) + result_cv2 * mask_binary
                composite = composite.astype(np.uint8)

                save_path = os.path.join(test_dir, f"{i:03d}.png")
                mask_save_path = os.path.join(gt_dir, f"{i:03d}_mask.png")
                cv2.imwrite(save_path, composite)
                cv2.imwrite(mask_save_path, final_anomaly_mask)

                if i % 20 == 0:
                    print(f"      💾 Saved: {i:03d}.png")

        if has_lora: pipe.unfuse_lora()

if __name__ == "__main__":
    main()