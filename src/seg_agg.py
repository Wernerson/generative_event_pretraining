import sys
sys.path.append("dinov2")
import torch
from torch import nn

from seg import SEG
from utils import spatiotemporal_aggregate


class SEGAgg(SEG):
    """
    Segmentation model variant that applies `spatiotemporal_aggregate` to
    encoder tokens computed from a short sequence of DSEC frames.
    """

    def __init__(self, config):
        super().__init__(config)
        if not hasattr(config, "frames_per_sample"):
            raise AttributeError("SegAgg requires `frames_per_sample` in config")
        self.frames_per_sample = config.frames_per_sample
        self.images_per_group = getattr(config, "images_per_group", self.frames_per_sample)

    def _encode_sequence(self, frames: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Encode consecutive event frames into patch tokens.

        Args:
            frames: Tensor of shape (B, frames_per_sample, C, H, W)

        Returns:
            tokens: (B, 2 * L, D) concatenated token sequence (current + next)
            tokens_per_image: number of tokens per individual event frame
        """
        if frames.ndim != 5 or frames.shape[1] != self.frames_per_sample:
            raise ValueError(
                f"Expected input with shape (B, {self.frames_per_sample}, C, H, W), got {frames.shape}"
            )

        event_curr = frames[:, 0]
        event_next = frames[:, 1]

        event_curr = self.pad(event_curr)
        event_next = self.pad(event_next)

        curr_tokens = self.event_encoder.forward_features(event_curr)["x_norm_patchtokens"]
        next_tokens = self.event_encoder.forward_features(event_next)["x_norm_patchtokens"]

        tokens_per_image = curr_tokens.shape[1]
        if next_tokens.shape[1] != tokens_per_image:
            raise ValueError("Event encoder produced mismatched token counts between frames")

        tokens = torch.cat((curr_tokens, next_tokens), dim=1)

        if hasattr(self.config, "n_tokens_per_image"):
            expected = self.config.n_tokens_per_image
            if expected != tokens_per_image:
                raise ValueError(
                    f"Encoder produced {tokens_per_image} tokens per image, "
                    f"but config expects {expected}"
                )
        return tokens, tokens_per_image

    def forward_transformer(self, x: torch.Tensor, ids: torch.Tensor | None = None):
        if isinstance(self.transformer, nn.Identity):
            return x
        B, T, C = x.shape
        if ids is None:
            ids = torch.full((B, T), 2, dtype=torch.int64, device=x.device)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.pos_embed(pos)
        modality_emb = self.transformer.modality_embed(ids)
        x_ = x + pos_emb + modality_emb
        for blk in self.transformer.blocks:
            x_ = blk(x_)
        x_ = self.transformer.norm(x_)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        if x.ndim != 5 or x.shape[1] != self.frames_per_sample:
            raise ValueError(
                f"Expected batched paired input with shape (B, {self.frames_per_sample}, C, H, W), got {x.shape}"
            )

        tokens, tokens_per_image = self._encode_sequence(x)
        B = tokens.shape[0]
        event_ids = torch.full((B, tokens_per_image), 2, dtype=torch.int64, device=tokens.device)
        next_event_ids = torch.full((B, tokens_per_image), 2, dtype=torch.int64, device=tokens.device)
        token_ids = torch.cat((event_ids, next_event_ids), dim=1)

        images_per_group = max(1, min(self.images_per_group, self.frames_per_sample))
        agg_tokens, agg_ids = spatiotemporal_aggregate(
            tokens,
            token_ids,
            tokens_per_image=tokens_per_image,
            images_per_group=images_per_group,
            valid_ids={2},
        )

        if not isinstance(self.transformer, nn.Identity):
            agg_tokens = self.forward_transformer(agg_tokens, ids=agg_ids)

        if agg_tokens.shape[1] < tokens_per_image:
            raise ValueError(
                f"Aggregated sequence too short ({agg_tokens.shape[1]}) "
                f"for decoder expecting {tokens_per_image} tokens."
            )

        patch_tokens = agg_tokens[:, :tokens_per_image]
        pred = self.forward_decoder(patch_tokens)
        loss = self.forward_loss(pred, y) if y is not None else None
        return pred, loss

    def visualize(self, frames, pred, labels, name, alpha=0.7, n=8):
        if frames.ndim == 5:
            frames = frames[:, -1]
        super().visualize(frames, pred, labels, name, alpha=alpha, n=n)


if __name__ == "__main__":
    from config import SegAggConfig

    config = SegAggConfig()
    model = SEGAgg(config).to(config.device)
    model.start()
    print("Training completed.")
