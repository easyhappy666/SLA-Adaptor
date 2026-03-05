names_1 = [
           'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
           'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
           'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight']

names_2 = [
    "1.to_k_ip.weight", "1.to_v_ip.weight", "3.to_k_ip.weight", "3.to_v_ip.weight", "5.to_k_ip.weight", "5.to_v_ip.weight", "7.to_k_ip.weight", "7.to_v_ip.weight", "9.to_k_ip.weight", "9.to_v_ip.weight", "11.to_k_ip.weight", "11.to_v_ip.weight", "13.to_k_ip.weight", "13.to_v_ip.weight", "15.to_k_ip.weight", "15.to_v_ip.weight", "17.to_k_ip.weight", "17.to_v_ip.weight", "19.to_k_ip.weight", "19.to_v_ip.weight", "21.to_k_ip.weight", "21.to_v_ip.weight", "23.to_k_ip.weight", "23.to_v_ip.weight", "25.to_k_ip.weight", "25.to_v_ip.weight", "27.to_k_ip.weight", "27.to_v_ip.weight", "29.to_k_ip.weight", "29.to_v_ip.weight", "31.to_k_ip.weight", "31.to_v_ip.weight"
]

mapping = {k: v for k, v in zip(names_1, names_2)}

import torch
from safetensors.torch import load_file
ckpt = "/output/checkpoint-4500/model.safetensors"
sd = load_file(ckpt)
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif "_ip." in k:
        ip_sd[mapping[k.replace("unet.", "")]] = sd[k]

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "/output/checkpoint-4500/ip_adapter.bin")