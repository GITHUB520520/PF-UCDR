import sys
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP
import math
import torch.nn as nn
from PIL import Image
from functools import reduce
from operator import mul
from src.utils import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
import numpy as np

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
_tokenizer = _Tokenizer()

import collections
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model: CLIP, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.tp_N_CTX
        dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.image_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=dtype).to(device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = ["a photo of " + name + " from " + "X " * n_ctx + "domain." for name in classnames]
        self.prefix_index = [length + 5 for length in name_lens]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("origin_text_embedding", embedding)

        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = [torch.cat([self.origin_text_embedding[i, :self.prefix_index[i]], ctx[i],
                              self.origin_text_embedding[i, self.prefix_index[i] + self.n_ctx:]], dim=0).view(1, -1,
                                                                                                              self.ctx_dim)
                   for i in range(self.n_cls)]
        prompts = torch.cat(prompts, dim=0)
        return prompts


# 新增自定义Transformer类（在image_encoder类外定义）
class TransformerWithIntermediate(nn.Module):
    def __init__(self, original_transformer):
        super().__init__()
        self.layers = original_transformer.resblocks  # 假设CLIP的transformer层存储在resblocks中
        self.num_layers = len(self.layers)

    def forward(self, x):
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        return x, hidden_states[-2]  # 返回最后一层和倒数第二层


class image_encoder(nn.Module):
    def __init__(self, clip_model: CLIP, cfg, dict_clss: dict, dict_doms: dict, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        # clip:CLIP = self.load_clip()
        self.conv1 = clip_model.visual.conv1
        width = self.conv1.out_channels
        self.feature_template = clip_model.visual.class_embedding
        self.feature_proj = clip_model.visual.proj
        patch_size = self.conv1.kernel_size
        self.clip_positional_embedding = clip_model.visual.positional_embedding
        self.generator = copy.deepcopy(clip_model.visual.transformer)
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.W_image = torch.nn.Linear(768, 768)
        self.W_prompt = torch.nn.Linear(768, 768)
        self.num_heads = 1
        self.temperature_dom = self.cfg.ratio_soft_dom
        self.temperature_cls = self.cfg.ratio_soft_cls

        self.ratio = self.cfg.ratio_prompt
        self.prompt = self.cfg.prompt

        self.layer_norm1 = torch.nn.LayerNorm(768)
        self.layer_norm2 = torch.nn.LayerNorm(768)
        # self.layer_norm_img = torch.nn.LayerNorm(768)
        # self.layer_norm_prm = torch.nn.LayerNorm(768)
        self.prompt_proj = torch.nn.Linear(768, 768)
        if self.cfg.DOM_PROJECT > -1:
            # only for prepend / add
            sp_dom_prompt_dim = self.cfg.DOM_PROJECT
            self.sp_dom_prompt_proj = nn.Linear(
                sp_dom_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_dom_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_dom_prompt_dim = width
            self.sp_dom_prompt_proj = nn.Identity()

        if self.cfg.CLS_PROJECT > -1:
            # only for prepend / add
            sp_cls_prompt_dim = self.cfg.CLS_PROJECT
            self.sp_cls_prompt_proj = nn.Linear(
                sp_cls_prompt_dim, width)
            nn.init.kaiming_normal_(
                self.sp_cls_prompt_proj.weight, a=0, mode='fan_out')
        else:
            sp_cls_prompt_dim = width
            self.sp_cls_prompt_proj = nn.Identity()

        # definition of specific prompts
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_dom_prompt_dim))  # noqa

        self.specific_domain_prompts = nn.Parameter(
            torch.randn(self.dom_num_tokens, sp_dom_prompt_dim))  # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_domain_prompts.data, -val, val)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + sp_cls_prompt_dim))  # noqa
        self.specific_class_prompts = nn.Parameter(
            torch.randn(self.cls_num_tokens, sp_cls_prompt_dim))  # layer, num_token, prompt_dim
        nn.init.uniform_(self.specific_class_prompts.data, -val, val)

    def incorporate_prompt(self, x, dom_index, cls_index, stage, img=None, dom_prompts=None, cls_prompts=None):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]  # batch size
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 65 768 49
        x = x.permute(0, 2, 1)  # 65 49 768
        base = self.cfg.GP_CLS_NUM_TOKENS + self.cfg.GP_DOM_NUM_TOKENS + 1
        # pdb.set_trace()
        if stage == 1:

            domain_prompts = torch.cat([
                self.sp_dom_prompt_proj(
                    self.specific_domain_prompts[self.prompt * (index):self.prompt * (index + 1)]
                ) for index in dom_index
            ], dim=0)
            class_prompts = torch.cat([
                self.sp_cls_prompt_proj(
                    self.specific_class_prompts[self.prompt * (index):self.prompt * (index + 1)]
                ) for index in cls_index
            ], dim=0)
            # pdb.set_trace()
            x = torch.cat((
                (self.feature_template + self.clip_positional_embedding[0]).expand(B, -1).view(B, 1, -1),
                domain_prompts.view(B, self.prompt, -1),
                class_prompts.view(B, self.prompt, -1),
                x + self.clip_positional_embedding[1:]
            ), dim=1)

        elif stage == 2:

            x = x + self.clip_positional_embedding[1:]

        elif stage == 3:

            sp_dom_prompts = self.sp_dom_prompt_proj(self.specific_domain_prompts)
            sp_cls_prompts = self.sp_cls_prompt_proj(self.specific_class_prompts)

            cls_prompt_mask = torch.zeros(B, sp_cls_prompts.shape[0], sp_cls_prompts.shape[1]).type(torch.bool).to(
                self.device)
            dom_prompt_mask = torch.zeros(B, sp_dom_prompts.shape[0], sp_dom_prompts.shape[1]).type(torch.bool).to(
                self.device)

            for i in range(B):
                start_idx = self.prompt * (cls_index[i])
                end_idx = self.prompt * (cls_index[i])
                cls_prompt_mask[i, start_idx:end_idx, :] = 1

            for i in range(B):
                start_idx = self.prompt * (dom_index[i])
                end_idx = self.prompt * (dom_index[i])
                dom_prompt_mask[i, start_idx:end_idx, :] = 1

            sp_cls_prompts = sp_cls_prompts.expand(B, -1, -1).masked_fill(cls_prompt_mask, 0)
            sp_dom_prompts = sp_dom_prompts.expand(B, -1, -1).masked_fill(dom_prompt_mask, 0)

            dom_attention_prompt = self.ImagePromptAttention_dom(img, sp_dom_prompts)
            cls_attention_prompt = self.ImagePromptAttention_cls(img, sp_cls_prompts)

            dom_prompts = torch.div(dom_attention_prompt, self.ratio)
            cls_prompts = torch.div(cls_attention_prompt, self.ratio)

            x = torch.cat(((
                                   self.feature_template + self.clip_positional_embedding[0]).expand(B, -1).view(B, 1,
                                                                                                                 -1),
                           dom_prompts,
                           cls_prompts,
                           x + self.clip_positional_embedding[1:]
                           ), dim=1)
        elif stage == 4:
            """
            test time generated prompts: no need mask specific prompts
            """
            dom_attention_prompt = self.ImagePromptAttention_dom(img, self.sp_dom_prompt_proj(
                self.specific_domain_prompts).expand(B, -1, -1))
            cls_attention_prompt = self.ImagePromptAttention_cls(img, self.sp_cls_prompt_proj(
                self.specific_class_prompts).expand(B, -1, -1))

            dom_prompts = torch.div(dom_attention_prompt, self.ratio)
            cls_prompts = torch.div(cls_attention_prompt, self.ratio)

            x = torch.cat(((
                                   self.feature_template + self.clip_positional_embedding[0]).expand(B, -1).view(B, 1,
                                                                                                                 -1),
                           dom_prompts,
                           cls_prompts,
                           x + self.clip_positional_embedding[1:]
                           ), dim=1)

        return x

    def vit(self, x, out_token):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)

        # 根据out_token选择使用transformer还是generator
        if out_token == 1:
            x = self.transformer(x)
        else:
            x = self.generator(x)

        # 获取最后一层的所有token输出
        last_layer_output = x.permute(1, 0, 2)  # (batch, seq_len, dim)

        if out_token == 1:
            # 原始的分类token输出
            cls_output = self.ln_post(last_layer_output[:, out_token, :])
            cls_output = cls_output @ self.feature_proj
            return cls_output, last_layer_output  # 返回分类特征和所有token特征
        else:
            # 其他情况
            output = self.ln_post(last_layer_output[:, :1, :])
            return output, last_layer_output  # 保持返回格式一致

    def forward(self, image, dom_id, cls_id, stage):
        if stage == 1:  # training for specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, stage)
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            output, last_layer_output = self.vit(x, 1)
            local_feature = self.ln_post(last_layer_output[:, 1:, :])
            local_feature = local_feature @ self.feature_proj
            return output, local_feature

        elif stage == 2:  # input: template + specific prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 2)  # cat template + specific prompts + image patch
            x = torch.nn.functional.dropout(x, self.cfg.dropout)
            output, _ = self.vit(x, 2)  # get genenrated prompts
            x = self.incorporate_prompt(image, dom_id, cls_id, 3, output)  # cat CLS generated prompts + image patch

            output, last_layer_output = self.vit(x, 1)
            local_feature = self.ln_post(last_layer_output[:, 1:, :])
            local_feature = local_feature @ self.feature_proj
            return output, local_feature

        elif stage == 4:
            x = self.incorporate_prompt(image, dom_id, cls_id, 2)
            output, _ = self.vit(x, 2)
            x = self.incorporate_prompt(image, dom_id, cls_id, 4, output)
            output, last_layer_output = self.vit(x, 1)
            local_feature = self.ln_post(last_layer_output[:, 1:, :])
            local_feature = local_feature @ self.feature_proj
            return output, local_feature

    def ImagePromptAttention_dom(self, images, prompts):
        images = self.W_image(images).chunk(self.num_heads, dim=-1)  # (batch_size, prompt_dim)
        prompts = self.W_prompt(prompts).chunk(self.num_heads, dim=-1)  # (batch_size, num_prompts, prompt_dim)
        combined_prompts = []

        for i in range(self.num_heads):
            attention_scores = torch.bmm(prompts[i], images[i].transpose(1, 2)) / self.temperature_dom
            attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_prompts, 1)
            # combined_prompts.append(torch.mm(attention_weights, prompts[i]))
            combined_prompts.append(torch.sum(attention_weights * prompts[i], dim=1))
        combined_prompts = torch.cat(combined_prompts, dim=-1).unsqueeze(1)
        combined_prompts = combined_prompts + self.layer_norm2(self.prompt_proj(combined_prompts))
        return combined_prompts

    def ImagePromptAttention_cls(self, images, prompts):

        images = self.W_image(images).chunk(self.num_heads, dim=-1)  # (batch_size, prompt_dim)
        prompts = self.W_prompt(prompts).chunk(self.num_heads, dim=-1)  # (batch_size, num_prompts, prompt_dim)
        combined_prompts = []

        for i in range(self.num_heads):
            attention_scores = torch.bmm(prompts[i], images[i].transpose(1, 2)) / self.temperature_cls
            attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_prompts, 1)
            # combined_prompts.append(torch.mm(attention_weights, prompts[i]))
            combined_prompts.append(torch.sum(attention_weights * prompts[i], dim=1))
        combined_prompts = torch.cat(combined_prompts, dim=-1).unsqueeze(1)
        combined_prompts = combined_prompts + self.layer_norm2(self.prompt_proj(combined_prompts))
        return combined_prompts


class RGBPhaseCrossAttentionGlobal(nn.Module):

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phase_attends_rgb = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.rgb_attends_phase = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_phase1 = nn.LayerNorm(embed_dim)
        self.norm_rgb1 = nn.LayerNorm(embed_dim)

        self.final_linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, rgb_features, phase_features):

        rgb_seq = rgb_features.unsqueeze(1)
        phase_seq = phase_features.unsqueeze(1)

        attn_output_phase, _ = self.phase_attends_rgb(
            query=phase_seq,  # (B, 1, E)
            key=rgb_seq,      # (B, 1, E)
            value=rgb_seq     # (B, 1, E) -> Use RGB info to update Phase
        )

        fused_phase_seq = self.norm_phase1(phase_seq + attn_output_phase)

        attn_output_rgb, _ = self.rgb_attends_phase(
            query=rgb_seq,      # (B, 1, E)
            key=phase_seq,      # (B, 1, E)
            value=phase_seq     # (B, 1, E) -> Use Phase info to update RGB
        )

        fused_rgb_seq = self.norm_rgb1(rgb_seq + attn_output_rgb)

        fused_rgb = fused_rgb_seq.squeeze(1)
        fused_phase = fused_phase_seq.squeeze(1)

        aggregated_feature = torch.cat([fused_rgb, fused_phase], dim=-1)

        aggregated_feature = self.final_linear(aggregated_feature)

        return aggregated_feature


class RGBPhaseCrossAttentionLayer(nn.Module):
    """
    接收RGB和Phase特征序列，返回经过一轮交互更新后的特征序列。
    设计用于可以堆叠。
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        # 两个独立的注意力层
        self.phase_attends_rgb = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.rgb_attends_phase = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 对应交叉注意力后的 Add & Norm
        self.norm_phase1 = nn.LayerNorm(embed_dim)
        self.norm_rgb1 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_features, phase_features):
        # Input shapes: [batch_size, seq_len, embed_dim]

        # Step 1: Phase Attends to RGB (Q=Phase, K=RGB, V=Phase)
        # Phase 特征根据其与 RGB 特征的相关性进行更新
        attn_output_phase, _ = self.phase_attends_rgb(
            query=phase_features,
            key=rgb_features,
            value=rgb_features
        )
        # Add & Norm 1 for Phase features
        fused_phase = self.norm_phase1(phase_features + attn_output_phase)

        # Step 2: RGB Attends to Phase (Q=RGB, K=Phase, V=RGB)
        # RGB 特征根据其与 Phase 特征的相关性进行更新
        attn_output_rgb, _ = self.rgb_attends_phase(
            query=rgb_features,
            key=phase_features,
            value=phase_features
        )
        # Add & Norm 1 for RGB features
        fused_rgb = self.norm_rgb1(rgb_features + attn_output_rgb)

        return fused_rgb, fused_phase


class StackedRGBPhaseCrossAttention(nn.Module):
    def __init__(self, num_layers=2, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RGBPhaseCrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_pool_rgb = nn.AdaptiveAvgPool1d(1)
        self.final_pool_phase = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, rgb_features, phase_features):
        for layer in self.layers:
            rgb_features, phase_features = layer(rgb_features, phase_features)

        # 经过多层交互后，再进行最终的聚合
        # (B, Seq, Dim) -> (B, Dim, Seq) -> (B, Dim, 1) -> (B, Dim)
        agg_rgb = self.final_pool_rgb(rgb_features.permute(0, 2, 1)).squeeze(-1)
        agg_phase = self.final_pool_phase(phase_features.permute(0, 2, 1)).squeeze(-1)
        # aggregated_feature = (agg_rgb + agg_phase) / 2

        aggregated_feature = torch.cat([agg_rgb, agg_phase], dim=-1)
        aggregated_feature = self.final_linear(aggregated_feature)
        return aggregated_feature


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),  # Dropout直接在Linear后使用
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)  # Dropout直接在Linear后使用
        )

    def forward(self, query_x, key_value_x, attn_mask=None):
        q_norm = self.norm1_q(query_x)
        kv_norm = self.norm1_kv(key_value_x)
        attn_out, _ = self.attn(query=q_norm, key=kv_norm, value=kv_norm, attn_mask=attn_mask)
        query_x = query_x + self.drop_path(attn_out)
        mlp_out = self.mlp(self.norm2(query_x))
        query_x = query_x + self.drop_path(mlp_out)
        return query_x


class FusingVisionEncoder(nn.Module):
    def __init__(self, clip_model: CLIP, num_fusion_layers=8, embed_dim=None, num_heads=12, cfg=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else {}

        visual_config = clip_model.visual

        self.embed_dim = embed_dim if embed_dim is not None else visual_config.conv1.out_channels
        _num_heads = num_heads if num_heads is not None else self.embed_dim // 64

        # 1. Patch Embedding (独立的 conv1 for RGB and Phase)
        self.conv1_rgb = copy.deepcopy(visual_config.conv1)
        self.conv1_phase = copy.deepcopy(visual_config.conv1)

        self.class_embedding_rgb = copy.deepcopy(visual_config.class_embedding)
        self.positional_embedding = copy.deepcopy(visual_config.positional_embedding)
        self.ln_pre = copy.deepcopy(visual_config.ln_pre)

        self.num_fusion_layers = num_fusion_layers
        self.transformer_self_attn_blocks = nn.ModuleList()
        self.cross_attention_blocks = nn.ModuleList()

        clip_transformer_blocks = visual_config.transformer.resblocks
        num_clip_layers = len(clip_transformer_blocks)

        for i in range(self.num_fusion_layers):
            layer_idx_to_copy = i % num_clip_layers
            self.transformer_self_attn_blocks.append(
                copy.deepcopy(clip_transformer_blocks[layer_idx_to_copy])
            )
            self.cross_attention_blocks.append(
                CrossAttentionBlock(
                    dim=self.embed_dim,
                    num_heads=_num_heads,
                    norm_layer=lambda dim_val: nn.LayerNorm(dim_val,
                                                            eps=getattr(clip_transformer_blocks[layer_idx_to_copy].ln_1,
                                                                        'eps', 1e-5)),
                    act_layer=type(clip_transformer_blocks[layer_idx_to_copy].mlp.act) if hasattr(
                        clip_transformer_blocks[layer_idx_to_copy].mlp, 'act') else nn.GELU
                )
            )

        self.ln_post = copy.deepcopy(visual_config.ln_post)

        # 7. Projection Head
        if hasattr(visual_config, 'proj') and visual_config.proj is not None:
            self.proj = copy.deepcopy(visual_config.proj)
        else:
            self.proj = nn.Identity()

        self._attn_mask_cache = {}  # 用于缓存掩码

    def _create_strict_corresponding_attn_mask(self, num_rgb_patches_in_query: int,
                                               num_phase_patches_in_kv: int,
                                               device: torch.device) -> torch.Tensor:
        mask_key = (num_rgb_patches_in_query, num_phase_patches_in_kv)
        if mask_key in self._attn_mask_cache:
            return self._attn_mask_cache[mask_key]

        N_q = 1 + num_rgb_patches_in_query  # Query: CLS + RGB patches
        N_kv = num_phase_patches_in_kv  # Key/Value: Phase patches

        attn_mask = torch.ones(N_q, N_kv, dtype=torch.bool, device=device)

        attn_mask[0, :] = False

        if num_rgb_patches_in_query == num_phase_patches_in_kv:
            for i in range(num_rgb_patches_in_query):
                attn_mask[i + 1, i] = False

        self._attn_mask_cache[mask_key] = attn_mask
        return attn_mask

    def forward(self, image_rgb, image_phase):
        B = image_rgb.shape[0]
        device = image_rgb.device

        # RGB处理: Patch投影 + Reshape + Permute
        # (B, C, H, W) -> (B, D, G, G) -> (B, D, N_p) -> (B, N_p, D)
        x_rgb_patches = self.conv1_rgb(image_rgb).reshape(B, self.embed_dim, -1).permute(0, 2, 1)
        num_actual_rgb_patches = x_rgb_patches.shape[1]

        # 添加CLS token
        cls_token_rgb = self.class_embedding_rgb.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        current_rgb_sequence = torch.cat([cls_token_rgb, x_rgb_patches], dim=1)  # (B, 1+N_p, D)

        # 添加位置编码 (修正后)
        # self.positional_embedding 形状是 (MaxSeqLen, D)
        # 我们需要 (1+N_p, D)
        pos_emb_rgb_slice_len = 1 + num_actual_rgb_patches
        pos_emb_rgb = self.positional_embedding[:pos_emb_rgb_slice_len, :]
        current_rgb_sequence = current_rgb_sequence + pos_emb_rgb.unsqueeze(0)  # (B, 1+N_p, D)
        current_rgb_sequence = self.ln_pre(current_rgb_sequence)

        # Phase处理: Patch投影 + Reshape + Permute
        x_phase_patches = self.conv1_phase(image_phase).reshape(B, self.embed_dim, -1).permute(0, 2, 1)
        num_actual_phase_patches = x_phase_patches.shape[1]

        # 添加位置编码 (修正后, 跳过CLS token的位置)
        pos_emb_phase_slice_len = num_actual_phase_patches
        pos_emb_phase = self.positional_embedding[1:(1 + pos_emb_phase_slice_len), :]
        current_phase_patches = x_phase_patches + pos_emb_phase.unsqueeze(0)  # (B, N_p_phase, D)
        current_phase_patches = self.ln_pre(current_phase_patches)

        # 创建注意力掩码
        strict_attn_mask = self._create_strict_corresponding_attn_mask(
            num_actual_rgb_patches,  # Query序列中patch的数量 (不含CLS)
            num_actual_phase_patches,  # Key/Value序列中patch的数量
            device
        )
        # Transformer 融合层
        for i in range(self.num_fusion_layers):
            # RGB 自注意力
            # (B, Seq, D) -> (Seq, B, D)
            rgb_for_self_attn = current_rgb_sequence.permute(1, 0, 2)
            rgb_after_self_attn = self.transformer_self_attn_blocks[i](rgb_for_self_attn)
            # (Seq, B, D) -> (B, Seq, D)
            current_rgb_sequence = rgb_after_self_attn.permute(1, 0, 2)

            # RGB 对 Phase 的掩码交叉注意力
            current_rgb_sequence = self.cross_attention_blocks[i](
                query_x=current_rgb_sequence,
                key_value_x=current_phase_patches,
                attn_mask=strict_attn_mask
            )

        # 提取全局和局部特征
        global_fused_rgb_feature = self.ln_post(current_rgb_sequence[:, 0, :])  # CLS token
        if not isinstance(self.proj, nn.Identity):
            global_fused_rgb_feature = global_fused_rgb_feature @ self.proj

        return global_fused_rgb_feature

class PF_UCDR(nn.Module):
    def __init__(self, cfg, dict_clss: dict, dict_doms: dict, device, isTest=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dict_clss = dict_clss
        self.dict_doms = dict_doms
        self.dom_num_tokens = len(self.dict_doms)
        self.cls_num_tokens = len(self.dict_clss)
        clip: CLIP = self.load_clip()
        self.image_encoder = image_encoder(clip, cfg, dict_clss, dict_doms, device)
        self.image_encoder_m = copy.deepcopy(image_encoder(clip, cfg, dict_clss, dict_doms, device))
        self.ratio_momentum = 0.999
        self.isTest = isTest
        self.stack_cross_attn_module = StackedRGBPhaseCrossAttention().to(device)
        self.global_attn_module = RGBPhaseCrossAttentionGlobal().to(device)
        self.fusing_vision_encoder = FusingVisionEncoder(
            clip_model=clip,
            num_fusion_layers=6,
            num_heads=8,
            cfg=cfg
        ).to(device)
        if self.cfg.tp_N_CTX != -1:
            self.text_prompt_learner = TextPromptLearner(self.cfg, self.dict_clss.keys(), clip, device)
            self.text_encoder = TextEncoder(clip)
            self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        else:
            self.text_encoder = clip.encode_text

        # 存放正负样本的队列，根据 class_id 进行分类
        self.cls_queues = {cls: collections.deque(maxlen=20) for cls in range(len(self.dict_clss))}
        # 存放相位图像特征的队列，根据 class_id 进行分类
        self.cls_ph_queues = {cls: collections.deque(maxlen=20) for cls in range(len(self.dict_clss))}

    def forward(self, image, phase, domain_name, class_name, stage):
        cls_id = utils.numeric_classes(class_name, self.dict_clss)
        dom_id = utils.numeric_classes(domain_name, self.dict_doms)

        if self.cfg.tp_N_CTX != -1:
            text_prompts = self.text_prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(text_prompts, tokenized_prompts)
        else:
            text_template = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(self.device)
            text_features = self.text_encoder(text_template)

        image_features, local_image_features = self.image_encoder(image, dom_id, cls_id, stage)  # batch, 512
        phase_features, local_phase_features = self.image_encoder(phase, dom_id, cls_id, stage)  # batch, 512

        combined_visual_feature = self.global_attn_module(image_features, phase_features)
        cross_image_feature = None
        cross_phase_feature = None
        fused_features =  self.fusing_vision_encoder(image, phase)
        fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)

        if stage == 1 or stage == 2:
            cross_image_feature = self.stack_cross_attn_module(local_image_features, local_phase_features)
            cross_image_feature = cross_image_feature / cross_image_feature.norm(dim=-1, keepdim=True)

        combined_visual_feature = combined_visual_feature / combined_visual_feature.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

        phase_features = phase_features / phase_features.norm(dim=-1, keepdim=True)
        local_phase_features = local_phase_features / local_phase_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 动量更新参数
        for param, cloned_param in zip(self.image_encoder.parameters(), self.image_encoder_m.parameters()):
            cloned_param.data = self.ratio_momentum * cloned_param.data + (1 - self.ratio_momentum) * param.data
        image_features_m, _ = self.image_encoder_m(image, dom_id, cls_id, stage)
        image_features_m = image_features_m / image_features_m.norm(dim=-1, keepdim=True)
        phase_features_m, _ = self.image_encoder_m(phase, dom_id, cls_id, stage)
        phase_features_m = phase_features_m / phase_features_m.norm(dim=-1, keepdim=True)
        for i in range(image_features_m.size(0)):
            current_cls = cls_id[i]
            self.cls_queues[current_cls].append(image_features_m[i])
            self.cls_ph_queues[current_cls].append(phase_features_m[i])
        queues = self.cls_queues
        ph_queues = self.cls_ph_queues
        if self.isTest:
            return image_features, phase_features, text_features, cls_id, dom_id, queues, ph_queues
        else:
            return image_features, phase_features, text_features, cls_id, dom_id, queues, ph_queues, local_image_features, local_phase_features, cross_image_feature, combined_visual_feature, fused_features

    def load_clip(self):
        backbone_name = self.cfg.clip_backbone
        print(f"=======load CLIP:{backbone_name}=========")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=self.device).eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location=self.device)

        model = clip.build_model(state_dict or model.state_dict())
        return model.float().to(self.device)