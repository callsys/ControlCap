import einops
import math
from functools import partial
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from lavis.common.dist_utils import download_cached_file
from lavis.models.eva_vit import VisionTransformer, interpolate_pos_embed, convert_weights_to_fp16

class ContextualVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.roi_size = (16, 16)
        self._roi_align = torchvision.ops.RoIAlign(output_size=self.roi_size,
                                                   spatial_scale=1 / 14,
                                                   sampling_ratio=1,
                                                   aligned=True
                                                   )

        self.cve_latent_dim = 512
        self.bridger_idxs = list(range(len(self.blocks)))[::-1][-10::2]
        self.cve_mlps = nn.ModuleList()
        for l in range(len(self.bridger_idxs)):
            cve_mlp = nn.Sequential(nn.Linear(self.embed_dim, self.cve_latent_dim),
                                    nn.GELU(),
                                    nn.Linear(self.cve_latent_dim, self.embed_dim))
            self.cve_mlps.append(cve_mlp)
            cve_mlp[-1].weight = nn.Parameter(torch.zeros_like(cve_mlp[-1].weight) * 0)
            cve_mlp[-1].bias = nn.Parameter(torch.zeros_like(cve_mlp[-1].bias) * 0)

    def seq2spatio(self, embeds):
        spatio_image_embeds = embeds[:, 1:]
        cls_embeds = embeds[:, 0][:, None]
        b, hw, c = spatio_image_embeds.shape
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
        spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return spatio_image_embeds, cls_embeds

    def spatio2seq(self, spatio_image_embeds, cls_embeds):
        b, c, h, w = spatio_image_embeds.shape
        spatio_image_embeds = spatio_image_embeds.permute(0, 2, 3, 1).reshape(b, -1, c)
        return torch.cat([cls_embeds, spatio_image_embeds], 1)

    def roi_align(self, image_embeds, samples):
        # prepare spatio image embeddings
        spatio_image_embeds, rois_cls_embeds = self.seq2spatio(image_embeds)

        # mask the padding bbox of the recognition task to save memory
        bboxes = samples["bboxes"]
        ids = samples["batch_idx"].to(torch.int64)
        rois = torch.cat([ids[:, None], bboxes], -1)

        # instance level feature encoder
        rois_embeds = self._roi_align(spatio_image_embeds, rois)
        rois_cls_embeds = rois_cls_embeds[ids]

        # back to sequence
        rois_embeds = self.spatio2seq(rois_embeds, rois_cls_embeds)
        return rois_embeds

    def cve_forward(self, embeds, samples, blk_idx):
        if blk_idx in self.bridger_idxs:
            bz = len(samples["image"])
            image_embeds = embeds[:bz]
            region_embeds = embeds[bz:]
            rois_embeds = self.roi_align(image_embeds, samples)
            mlp = self.cve_mlps[self.bridger_idxs.index(blk_idx)]
            embeds = torch.cat([mlp(rois_embeds - region_embeds) + region_embeds, image_embeds], 0)
        return embeds

    def forward(self, x, samples):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
                x = self.cve_forward(x, samples, blk_idx)

        bz = len(samples["image"])
        return x[bz:]

def create_cve_vit_g(img_size=224, drop_path_rate=0.4, use_checkpoint=False, precision="fp16"):
    model = ContextualVisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408 // 88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
    )
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")
    interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    #     print(incompatible_keys)

    if precision == "fp16":
        #         model.to("cuda")
        convert_weights_to_fp16(model)
    return model

