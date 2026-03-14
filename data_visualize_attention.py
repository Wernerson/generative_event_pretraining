import argparse
import os
import sys
sys.path.append("dinov2")

import torchvision
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

from dinov2.models.vision_transformer import vit_base

NIMA_ME = [0.9673029496145361, 0.929740832760733, 0.9624378831461544]
NIMA_SE = [0.12036860792540319, 0.1674319634709885, 0.12649585555644863]
valid_transform      = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(NIMA_ME, NIMA_SE),
                                transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
                                ])
def main(args):
    # Model setup
    model = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6, num_register_tokens=4)
    patch_size = model.patch_size
    num_heads = model.num_heads
    params = torch.load(args.pretrained_backbone_path, weights_only=True)
    if "event_encoder" in params.keys():
        params = params["event_encoder"]
    if "encoder" in params.keys():
        params = params["encoder"]
    # params = torch.load(args.pretrained_backbone_path, weights_only=True)["event_encoder"]
    model.load_state_dict(params, strict=True)
    model = model.to(args.device)
    model.eval()

    # Data preparing
    def resize_img(img, patch_size):
        w, h = img.size
        if max(w, h) > 1400:
            ratio = 1400 / max(w, h)
            w, h = int(w * ratio), int(h * ratio)
            w, h = w // patch_size * patch_size, h // patch_size * patch_size
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        else:
            new_w = 1 * w // patch_size * patch_size
            new_h = 1 * h // patch_size * patch_size
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
        return img
    image = Image.open(args.image_dir)
    # image = resize_img(image, patch_size)
    image_tensor = valid_transform(image).unsqueeze(0).to(args.device)
    print(f"image shape: {image_tensor.shape}")
    h, w = image_tensor.shape[2:]
    torchvision.utils.save_image(image_tensor, os.path.join(args.output_dir, "image.png"), normalize=True)

    # Attention map visualization
    def save_colormap_image(tensor, path, cmap_name='viridis'):
        array = tensor.detach().cpu().numpy()
        array = (array - array.min()) / (array.max() - array.min())
        colormap = plt.get_cmap(cmap_name)
        colored = colormap(array)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        Image.fromarray(colored).save(path)
    h_attention_map, w_attention_map = h // patch_size, w // patch_size
    attentions = model.get_last_self_attention(image_tensor.to(args.device))
    print(f"attentions shape: {attentions.shape}")
    attentions = attentions[0, :, 0, 5:].reshape(num_heads, -1)
    attentions = attentions.reshape(num_heads, h_attention_map, w_attention_map)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
    for j in range(num_heads):
        save_colormap_image(attentions[j].squeeze(), os.path.join(args.output_dir, f"attn-head{j}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/data/storage/jianwen/N_ImageNet/extracted_val/n03710193/ILSVRC2012_val_00000364.png", type=str)
    parser.add_argument("--pretrained_backbone_path", default="/data/storage/jianwen/cache/dinov2/dinov2_vitb14_reg4_pretrain.pth", type=str)
    # parser.add_argument("--pretrained_backbone_path", default="/data/storage/jianwen/cache/ckpts/2025-11-07-20:15_cls/epoch2400_0.9955.pt", type=str)
    parser.add_argument("--device", default="cuda:2", type=str, help="Device to run the model, e.g. 'cuda:2'")
    args = parser.parse_args()
    
    args.output_dir = f"attention_maps/{os.path.basename(args.pretrained_backbone_path).split('.')[0]}_{os.path.basename(args.image_dir).split('.')[0]}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"attention maps will be saved to {args.output_dir}")
    
    main(args)