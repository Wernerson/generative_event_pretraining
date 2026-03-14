import sys
sys.path.append("dinov2")
# --------------------------------deterministic setting-------------------------------- #
import numpy as np
seed = 0
np.random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from torch import nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
g = torch.Generator()
g.manual_seed(seed)
torch.use_deterministic_algorithms(True)
# --------------------------------------------------------------------------------------- #
from datetime import datetime
import os
import time
import einops
import torchvision
from torch import nn
import torch
import torch.nn.functional as F
from torchinfo import summary
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small
from model import Block, Transformer
from utils import FlowMetrics, get_lr, get_param_groups, masked_l1_loss, FlowMetrics

from torch.utils.tensorboard import SummaryWriter

def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    """
    Visualizes optical flow using the HSV color space.
    
    Args:
        flow (torch.Tensor): Optical flow tensor of shape [B, 2, H, W].
    
    Returns:
        torch.Tensor: RGB visualization of shape [B, 3, H, W] with values in [0, 1].
    """
    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"Input flow must be a torch.Tensor, got {type(flow)}")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Input flow must have shape [B, 2, H, W], got {flow.shape}")

    B, _, H, W = flow.shape
    device = flow.device

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    # Calculate magnitude and angle
    magnitude = torch.sqrt(u**2 + v**2)  # Shape: [B, H, W]
    angle = torch.atan2(v, u)            # Shape: [B, H, W]

    # --- FIX STARTS HERE ---
    
    # Reshape magnitude to (B, H*W) to find max over spatial dimensions
    mag_view = magnitude.view(B, -1)
    
    # Find the max magnitude for each sample in the batch
    # mag_max will have shape [B, 1]
    mag_max = torch.max(mag_view, dim=1, keepdim=True)[0]

    # Reshape mag_max to [B, 1, 1] so it can be broadcasted to [B, H, W]
    mag_max_reshaped = mag_max.view(B, 1, 1)

    # Avoid division by zero by adding a small epsilon
    # Broadcasting happens here: magnitude [B, H, W] / mag_max_reshaped [B, 1, 1]
    normalized_magnitude = magnitude / (mag_max_reshaped + 1e-6)

    # --- FIX ENDS HERE ---

    # Map angle to Hue [0, 1]
    hue = (angle + np.pi) / (2 * np.pi)
    
    saturation = torch.ones(B, H, W, device=device)
    
    value = normalized_magnitude

    # Stack to create an HSV image
    hsv = torch.stack([hue, saturation, value], dim=3)  # [B, H, W, 3]

    # --- Vectorized HSV to RGB conversion (no changes needed here) ---
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hi = torch.floor(h * 6.0) % 6.0
    f = (h * 6.0) - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    hi = hi.long()
    
    # This part is complex but correct. It builds the RGB channels based on the hi index.
    # It can be simplified for readability, but let's keep it for performance.
    v_exp = v.unsqueeze(-1)
    t_exp = t.unsqueeze(-1)
    p_exp = p.unsqueeze(-1)
    q_exp = q.unsqueeze(-1)
    
    rgb_exp = torch.cat([v_exp, q_exp, p_exp, p_exp, t_exp, v_exp], dim=-1) # [B, H, W, 6]
    
    # Create mask for each of the 6 cases of hi
    hi_exp = hi.unsqueeze(-1) # [B, H, W, 1]
    mask = torch.zeros(B, H, W, 6, device=device).scatter_(-1, hi_exp, 1) # [B, H, W, 6]
    
    # Select the correct triplet for each pixel
    selected_triplets = rgb_exp * mask # [B, H, W, 6]
    
    # Build the final RGB channels by summing up the selected components
    r = selected_triplets[..., 0] + selected_triplets[..., 5]
    g = selected_triplets[..., 1] + selected_triplets[..., 2]
    b = selected_triplets[..., 3] + selected_triplets[..., 4]
    
    rgb = torch.stack([r, g, b], dim=3)
    
    return rgb.permute(0, 3, 1, 2)

class OPT(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6)
        if config.event_encoder_weight is not None:
            self.event_encoder.load_state_dict(config.event_encoder_weight, strict=True)
            print("*" * 50 + " event encoder loaded")
        self.transformer    = nn.ModuleDict(dict(   modality_embed = nn.Embedding(4, config.n_embed),
                                                    pos_embed = nn.Embedding(config.window_size, config.n_embed),
                                                    blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
                                                    norm = nn.LayerNorm(config.n_embed),
                                                ))if hasattr(config, "transformer_weight") else nn.Identity()
        if hasattr(config, "transformer_weight") and config.transformer_weight is not None:
            self.transformer.load_state_dict(config.transformer_weight, strict=True)
            print("*" * 50 + " transformer loaded")
        self.decoder = nn.Linear(config.n_embed, 2 * config.P ** 2)
        
        # training related
        self.train_dataloader       = torch.utils.data.DataLoader(self.config.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.valid_dataloader       = torch.utils.data.DataLoader(self.config.valid_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.n_workers, pin_memory=True, drop_last=False)
        self.amp = torch.amp.autocast(device_type = "cuda")
        self.scaler = torch.amp.GradScaler(device = "cuda")
        self.optimizer = torch.optim.AdamW(get_param_groups(self, self.config.wd, self.config.encoder_lr_mult, self.config.transformer_lr_mult))
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.flow = FlowMetrics()

        if self.config.encoder_frozen:
            print("freeze event encoder")
            for name, param in self.named_parameters():
                if "decoder" not in name and "transformer" not in name:
                    param.requires_grad = False

        os.makedirs("src/runs", exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"src/runs/{self.now}_opt")
            
    def forward_encoder(self, x):
        return self.event_encoder.forward_features(x)["x_norm_patchtokens"]
    
    def forward_transformer(self, x):
        B, T, C = x.shape
        ids = torch.ones(B, T, dtype=torch.int64, device=x.device) * 2  # 1 for image, 2 for event
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.pos_embed(pos)
        modality_emb = self.transformer.modality_embed(ids)
        x_ = x + pos_emb + modality_emb
        for i, blk in enumerate(self.transformer.blocks):
            x_ = blk(x_)
        # x_ = self.transformer.norm(x_)
        return x_ + x

    def forward_decoder(self, x):
        x = self.decoder(x)
        x = einops.rearrange(x, "b (l1 l2) (c p1 p2) -> b c (l1 p1) (l2 p2)", c=2, p1=self.config.P, p2=self.config.P, l1=self.config.H // self.config.P, l2=self.config.W // self.config.P)
        return x

    def forward_loss(self, pred, target):
        # loss = masked_l1_loss(pred, target[:, :2], target[:, 2:])
        loss = F.l1_loss(pred, target[:, :2])
        return loss

    def forward(self, x, y=None):
        z = self.forward_encoder(x)
        if hasattr(config, "transformer_weight"):
            z = self.forward_transformer(z) 
        pred = self.forward_decoder(z)
        loss = self.forward_loss(pred, y) if y is not None else None
        return pred, loss

    @torch.no_grad()
    def visualize(self, image, pred_flow, target, name, alpha=0.5, n=8):
        """
        Generates and saves a visualization grid for optical flow results.

        Args:
            image (torch.Tensor): The original input image batch. Shape: [B, C, H, W].
            pred_flow (torch.Tensor): The predicted optical flow. Shape: [B, 2, H, W].
            label_flow (torch.Tensor): The ground-truth optical flow. Shape: [B, 2, H, W].
            valid_mask (torch.Tensor): The validity mask for the ground-truth. Shape: [B, 1, H, W].
            name (str): The filename for the output image.
            alpha (float): The blending factor for the overlay.
            n (int): The number of samples from the batch to visualize.
        """
        # --- 1. Un-normalize the original image ---
        # Ensure M and S are on the correct device
        # print(image.device, pred_flow.device, target.device)
        label_flow, valid_mask = target[:, :2], target[:, 2:]
        M = torch.tensor(self.config.ME, device=image.device)[None, :, None, None]
        S = torch.tensor(self.config.SE, device=image.device)[None, :, None, None]
        orig = torch.clamp(image * S + M, 0, 1).to(image.device) # [B, 3, H, W]

        # --- 2. Convert flow fields to RGB color maps ---
        pred_rgb = flow_to_rgb(pred_flow)    # [B, 3, H, W]
        label_rgb = flow_to_rgb(label_flow)  # [B, 3, H, W]

        # --- 3. Mask out invalid pixels in the ground-truth visualization ---
        # Invalid areas will become black
        label_rgb = label_rgb * valid_mask

        # --- 4. Create blended overlay of prediction on original image ---
        blend = orig.mul(1 - alpha) + pred_rgb.mul(alpha)

        # --- 5. Build the visualization grid: [Original | Prediction | Blend | Ground Truth] ---
        # We need to slice the first n samples from each tensor
        # Ensure all tensors are float for make_grid
        tensors_to_grid = [
            orig[:n], 
            pred_rgb[:n], 
            blend[:n], 
            label_rgb[:n]
        ]
        
        # Handle cases where the input image might not be 3 channels (e.g., event image)
        for i, t in enumerate(tensors_to_grid):
            if t.shape[1] == 1: # if it's a single channel image
                tensors_to_grid[i] = t.repeat(1, 3, 1, 1) # convert to 3-channel grayscale
        
        rendered = torchvision.utils.make_grid(
            torch.cat(tensors_to_grid),
            nrow=n
        )

        # --- 6. Save the final image ---
        torchvision.utils.save_image(rendered, f"src/{name}.png")
        print(f"Saved visualization to src/{name}.png")
    
    def train_step(self, x, y, global_step):
        x, y = x.to(self.config.device), y.to(self.config.device)
        t0 = time.time()
        self.train()
        current_lr = get_lr(global_step, self.config.warmup_steps, self.config.lr, self.config.steps, self.config.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr * param_group["lr_mult"]
            
        with self.amp:
            pred, loss = self.forward(x, y)
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=1.,)
        nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        t1 = time.time()
        if (global_step + 1) % self.config.log_every == 0:
            self.writer.add_scalar("loss/train", loss.item(), global_step + 1)
            dt = (t1 - t0)
            self.visualize(x, pred, y, name="vis_opt_train")
            num_tokens_per_secend =  self.config.batch_size * self.config.n_tokens_per_image / dt
            print(f"step: {global_step + 1}, lr: {current_lr :.8f}, loss: {loss.item() :.4f}, grad_norm: {grad_norm:.4f}, input: {x.shape}, dt:{dt: .2f}, throughput: {num_tokens_per_secend :.2f} t/s")
        
    def start(self):
        train_iter = iter(self.train_dataloader)
        # self.validate(0)
        for step in range(self.config.steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                x, y = next(train_iter)

            if step == 0:
                # print(x.shape, y.shape)
                # print((y[:, -1]==0).sum(), (y[:, -1]==1).sum())
                # torchvision.utils.save_image(y[:, -1:].to(dtype=torch.float32), "src/opt_input.png")
                # exit()
                summary(self, input_data=(x, y), device=self.config.device, depth=2)
            self.train_step(x, y, step)

            if (step + 1) % self.config.valid_every == 0 or (step + 1) == self.config.steps:
                self.validate(step)
    
    @ torch.no_grad()
    def validate(self, step):
        self.eval()
        valid_loss = 0.0
        self.flow.reset()
        for i, (x, y) in enumerate(tqdm(self.valid_dataloader)):
            x, y = x.to(self.config.device), y.to(self.config.device)
            pred, loss = self.forward(x, y)
            valid_loss += loss.item()
            self.flow.update(pred, y)
            if i >= 8 and step != 0 and (step+1) != self.config.steps:
                break
        self.visualize(x, pred, y, name="vis_opt_valid")
        flow_dict = self.flow.compute()

        valid_loss /= (i+1)
        
        print(f"step: {step}, valid loss: {valid_loss:.4f}")
        print("flow_dict:", {key: round(value, 4) for key, value in flow_dict.items()})
        
        self.writer.add_scalar("loss/valid", valid_loss, step)
        self.writer.add_scalar("epe/valid", flow_dict["epe"], step)
        self.writer.add_scalar("ae/valid", flow_dict["ae"], step)
        self.writer.add_scalar("1pe/valid", flow_dict["1pe"], step)
        self.writer.add_scalar("2pe/valid", flow_dict["2pe"], step)
        self.writer.add_scalar("3pe/valid", flow_dict["3pe"], step)
        
        param_dict = {
            "event_encoder": self.event_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": step,}
        model_save_path = f"/data/storage/jianwen/cache/ckpts/{self.now}_opt"
        os.makedirs(model_save_path, exist_ok=True)
        print(f"------------------------- saving model to: {model_save_path}")
        torch.save(param_dict, os.path.join(model_save_path, f"epoch{step + 1}_{valid_loss:.4f}.pt"))
        
if __name__ == "__main__":
    from config import OPTConfig
    config = OPTConfig()
    model = OPT(config).to(config.device)
    model.start()
    print("Training completed.")