import math

import comm
import torch
from easydict import EasyDict
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_

from Utils.pointnet_util import PointNetFeaturePropagation
from Models.Build_Model import models
from torch import nn
from functools import partial
from Utils.Tool import index_points
from Utils import Misc
from Models.mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        group_rate,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        **factory_kwargs
):
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        group_rate,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class Bi_Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            group_rate,
            layer_idx,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            residual_in_fp32=False,
            fused_add_norm=False,
            drop_path: int = 0.1,
            **factory_kwargs
    ):
        super(Bi_Mamba, self).__init__()
        self.pos_dir, self.neg_dir = [
            create_block(
                d_model,
                group_rate,
                layer_idx=layer_idx,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                drop_path=drop_path,
                **factory_kwargs,
            ) for _ in range(2)
        ]

    def forward(self, hidden_states, inference_params=None):
        return self.neg_dir(
            self.pos_dir(hidden_states, inference_params=inference_params)[0].flip(dims=(-2,)),
            inference_params=inference_params)[0].flip(dims=(-2,))


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            group_rate: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ):
        super(MixerModel, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                Bi_Mamba(
                    d_model,
                    group_rate,
                    layer_idx=i,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    drop_path=drop_path,
                    **factory_kwargs
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class Group(nn.Module):
    def __init__(self, group_size, num_group=None, is_first=False):
        super().__init__()
        self.is_first = is_first
        self.group_size = group_size
        self.num_group = num_group
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = Misc.fps(xyz.contiguous(), self.num_group) if not self.is_first else xyz  # B G 3
        # knn to get the neighborhood
        # import ipdb; ipdb.set_trace()
        # idx = knn_query(xyz, center, self.group_size)  # B G M
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group if not self.is_first else xyz.shape[1]
        assert idx.size(2) == self.group_size
        return idx, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(encoder_channel // 2, encoder_channel, 1),
            nn.BatchNorm1d(encoder_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(encoder_channel, encoder_channel, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(encoder_channel * 2, encoder_channel * 2, 1),
            nn.BatchNorm1d(encoder_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(encoder_channel * 2, encoder_channel, 1)
        )

    def forward(self, feature_groups):
        '''
            feature_groups : B G N D
            -----------------
            feature_global : B G C
        '''
        bs, g, n, d = feature_groups.shape
        feature_groups = feature_groups.reshape(bs * g, n, d)
        # encoder
        feature = self.first_conv(feature_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, -1)


class Down_Sampling(nn.Module):
    def __init__(self, d_model, group_size, num_group, is_first):
        super(Down_Sampling, self).__init__()
        self.group_divider = Group(group_size=group_size, num_group=num_group, is_first=is_first)
        self.encoder = Encoder(d_model)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, xyz, features):
        idx, center = self.group_divider(xyz)
        # center point features plus pos info
        group_input = self.encoder(index_points(features, idx)) + self.pos_embed(center)
        return center, group_input


class Up_Sampling(nn.Module):
    def __init__(self, dim, is_first):
        super(Up_Sampling, self).__init__()
        self.is_first = is_first

        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        if not self.is_first:
            self.fc1 = nn.Sequential(
                nn.Linear(2 * dim, dim),
                SwapAxes(),
                nn.BatchNorm1d(dim),  # TODO
                SwapAxes(),
                nn.ReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(dim, dim),
                SwapAxes(),
                nn.BatchNorm1d(dim),  # TODO
                SwapAxes(),
                nn.ReLU(),
            )
            self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        if not self.is_first:
            feats1 = self.fc1(points1)
            feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
            feats2 = self.fc2(points2)
            return xyz2, feats1 + feats2
        else:
            return xyz1, points1


class Order_Merging(nn.Module):
    def __init__(
            self,
            d_model,
            group_rate,
            n_layer,
            order_prompt: bool,
            rms_norm,
            drop_out_in_block,
            drop_path,
    ):
        super(Order_Merging, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.order_prompt = order_prompt
        if self.order_prompt:
            self.z_order, self.y_order, self.x_order = nn.Parameter(
                trunc_normal_(torch.zeros(3, 1, d_model), std=0.02)
            ).to(device).chunk(3, dim=0)
            self.z_order_pos, self.y_order_pos, self.x_order_pos = nn.Parameter(
                trunc_normal_(torch.zeros(3, 1, d_model), std=0.02)
            ).to(device).chunk(3, dim=0)

        self.mixer_z, self.mixer_y, self.mixer_x = [
            MixerModel(
                d_model,
                group_rate,
                n_layer,
                rms_norm=rms_norm,
                drop_out_in_block=drop_out_in_block,
                drop_path=drop_path
            ) for _ in range(3)
        ]

        self.down = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, center, group_input):

        center_z = center[:, :, 2].argsort(dim=-1)
        center_y = center[:, :, 1].argsort(dim=-1)
        center_x = center[:, :, 0].argsort(dim=-1)

        group_input_z = index_points(group_input, center_z)
        group_input_y = index_points(group_input, center_y)
        group_input_x = index_points(group_input, center_x)

        if self.order_prompt:
            group_input_z = torch.cat([(self.z_order + self.z_order_pos).expand(center.shape[0], -1, -1), group_input_z], dim=1)
            group_input_y = torch.cat([(self.y_order + self.y_order_pos).expand(center.shape[0], -1, -1), group_input_y], dim=1)
            group_input_x = torch.cat([(self.x_order + self.x_order_pos).expand(center.shape[0], -1, -1), group_input_x], dim=1)

        mixer_z, mixer_y, mixer_x = self.mixer_z(group_input_z), self.mixer_y(group_input_y), self.mixer_x(group_input_x)

        if self.order_prompt:
            mixer_z, mixer_y, mixer_x = mixer_z[:, 1:], mixer_y[:, 1:], mixer_x[:, 1:]

        # retrieve index
        center_z_retrieve, center_y_retrieve, center_x_retrieve = center_z.argsort(dim=-1), center_y.argsort(dim=-1), center_x.argsort(dim=-1)
        group_output = torch.cat(
            [index_points(mixer_z, center_z_retrieve),
             index_points(mixer_y, center_y_retrieve),
             index_points(mixer_x, center_x_retrieve)], dim=-1)
        return center, self.down(group_output)


@models.register_module()
class Cloud_Mamba_Reg(nn.Module):
    def __init__(self, cfgs_model):
        super(Cloud_Mamba_Reg, self).__init__()
        self.cfgs_model = cfgs_model

        self.input_dim = cfgs_model.input_dim
        self.trans_dim = cfgs_model.trans_dim
        self.group_rate = cfgs_model.group_rate
        self.enc_layers = cfgs_model.encoder_layers
        self.order_prompt = cfgs_model.order_prompt
        self.cls_dim = cfgs_model.cls_dim

        self.rms_norm = False if not hasattr(cfgs_model, "rms_norm") else cfgs_model.rms_norm
        self.drop_out_in_block = 0. if not hasattr(cfgs_model, "drop_out_in_block") else cfgs_model.drop_out_in_block
        self.drop_path = 0. if not hasattr(cfgs_model, "drop_path") else cfgs_model.drop_path

        self.group_size = cfgs_model.group_size
        self.num_group = cfgs_model.num_group

        self.emb = nn.Sequential(
            nn.Linear(self.input_dim, self.trans_dim),
            nn.ReLU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        self.down_sampling, self.order_merging_enc = nn.ModuleList(), nn.ModuleList()
        for i, n_layer in enumerate(self.enc_layers):
            self.down_sampling.append(
                Down_Sampling(self.trans_dim * (2 ** i),
                              self.group_size,
                              self.num_group // (4 ** i),
                              i == 0)
            )
            self.order_merging_enc.append(
                Order_Merging(self.trans_dim * (2 ** i),
                              self.group_rate,
                              n_layer,
                              self.order_prompt,
                              self.rms_norm,
                              self.drop_out_in_block,
                              self.drop_path)
            )

        self.output_dim = self.trans_dim * (2 ** (len(self.enc_layers) - 1))
        self.norm = nn.LayerNorm(self.output_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, self.cls_dim)
        )

        self.apply(self._init_weights)

    def forward(self, xyz):
        features = self.emb(xyz)
        xyz = xyz[..., 0:3]

        for i in range(len(self.enc_layers)):
            center, group_input = self.down_sampling[i](xyz, features) if i != 0 else (xyz, features)
            xyz, features = self.order_merging_enc[i](center, group_input)

        x = self.norm(features).mean(1)
        logit = self.cls_head(x)
        return logit

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@models.register_module()
class Cloud_Mamba_Part(nn.Module):
    def __init__(self, cfgs_model):
        super(Cloud_Mamba_Part, self).__init__()
        self.cfgs_model = cfgs_model

        self.input_dim = cfgs_model.input_dim
        self.trans_dim = cfgs_model.trans_dim
        self.group_rate = cfgs_model.group_rate
        self.enc_layers = cfgs_model.encoder_layers
        self.dec_layers = cfgs_model.decoder_layers
        self.order_prompt = cfgs_model.order_prompt
        self.cls_dim = cfgs_model.cls_dim
        self.cate = cfgs_model.category

        self.rms_norm = False if not hasattr(cfgs_model, "rms_norm") else cfgs_model.rms_norm
        self.drop_out_in_block = 0. if not hasattr(cfgs_model, "drop_out_in_block") else cfgs_model.drop_out_in_block
        self.drop_path = 0. if not hasattr(cfgs_model, "drop_path") else cfgs_model.drop_path

        self.group_size = cfgs_model.group_size
        self.num_group = cfgs_model.num_group

        self.emb = nn.Sequential(
            nn.Linear(self.input_dim, self.trans_dim),
            nn.ReLU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        self.down_sampling, self.order_merging_enc = nn.ModuleList(), nn.ModuleList()
        for i, n_layer in enumerate(self.enc_layers):
            self.down_sampling.append(
                Down_Sampling(self.trans_dim * (2 ** i),
                              self.group_size,
                              self.num_group // (4 ** i),
                              i == 0)
            )
            self.order_merging_enc.append(
                Order_Merging(self.trans_dim * (2 ** i),
                              self.group_rate,
                              n_layer,
                              self.order_prompt,
                              self.rms_norm,
                              self.drop_out_in_block,
                              self.drop_path)
            )

        self.output_dim = self.trans_dim * (2 ** (len(self.enc_layers) - 1))
        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

        self.up_sampling, self.order_merging_dec = nn.ModuleList(), nn.ModuleList()
        for i, n_layer in enumerate(self.dec_layers):
            self.up_sampling.append(
                Up_Sampling(self.output_dim // (2 ** i),
                            i == 0)
            )
            self.order_merging_dec.append(
                Order_Merging(self.output_dim // (2 ** i),
                              self.group_rate,
                              n_layer,
                              self.order_prompt,
                              self.rms_norm,
                              self.drop_out_in_block,
                              self.drop_path)
            )

        self.norm = nn.LayerNorm(self.trans_dim + self.cate)

        self.cls_head = nn.Sequential(
            nn.Conv1d(self.trans_dim + self.cate, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, self.cls_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, xyz, cat_prompt):
        features = self.emb(xyz)
        xyz = xyz[..., 0:3]
        # encoding
        xyz_and_feats = []
        for i in range(len(self.enc_layers)):
            center, group_input = self.down_sampling[i](xyz, features) if i != 0 else (xyz, features)
            xyz, features = self.order_merging_enc[i](center, group_input)
            xyz_and_feats.append((xyz, features))

        # decoding
        xyz = xyz_and_feats[-1][0]
        features = self.fc(xyz_and_feats[-1][1])
        for i in range(len(self.dec_layers)):
            center, group_input = self.up_sampling[i](xyz, features, xyz_and_feats[- i - 1][0], xyz_and_feats[- i - 1][1])
            xyz, features = self.order_merging_dec[i](center, group_input)

        cat_prompt = cat_prompt.reshape(-1, 1, self.cate).repeat(1, self.num_group, 1)
        x = self.norm(torch.cat([features, cat_prompt], -1))
        logit = self.cls_head(x.transpose(2, 1))
        return logit.transpose(2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@models.register_module()
class Cloud_Mamba_Sem(nn.Module):
    def __init__(self, cfgs_model):
        super(Cloud_Mamba_Sem, self).__init__()
        self.cfgs_model = cfgs_model

        self.input_dim = cfgs_model.input_dim
        self.trans_dim = cfgs_model.trans_dim
        self.group_rate = cfgs_model.group_rate
        self.enc_layers = cfgs_model.encoder_layers
        self.dec_layers = cfgs_model.decoder_layers
        self.order_prompt = cfgs_model.order_prompt
        self.cls_dim = cfgs_model.cls_dim

        self.rms_norm = False if not hasattr(cfgs_model, "rms_norm") else cfgs_model.rms_norm
        self.drop_out_in_block = 0. if not hasattr(cfgs_model, "drop_out_in_block") else cfgs_model.drop_out_in_block
        self.drop_path = 0. if not hasattr(cfgs_model, "drop_path") else cfgs_model.drop_path

        self.group_size = cfgs_model.group_size
        self.num_group = cfgs_model.num_group

        self.emb = nn.Sequential(
            nn.Linear(self.input_dim, self.trans_dim),
            nn.ReLU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        self.down_sampling, self.order_merging_enc = nn.ModuleList(), nn.ModuleList()
        for i, n_layer in enumerate(self.enc_layers):
            self.down_sampling.append(
                Down_Sampling(self.trans_dim * (2 ** i),
                              self.group_size,
                              self.num_group // (4 ** i),
                              i == 0)
            )
            self.order_merging_enc.append(
                Order_Merging(self.trans_dim * (2 ** i),
                              self.group_rate,
                              n_layer,
                              self.order_prompt,
                              self.rms_norm,
                              self.drop_out_in_block,
                              self.drop_path)
            )

        self.output_dim = self.trans_dim * (2 ** (len(self.enc_layers) - 1))
        self.fc = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

        self.up_sampling, self.order_merging_dec = nn.ModuleList(), nn.ModuleList()
        for i, n_layer in enumerate(self.dec_layers):
            self.up_sampling.append(
                Up_Sampling(self.output_dim // (2 ** i),
                            i == 0)
            )
            self.order_merging_dec.append(
                Order_Merging(self.output_dim // (2 ** i),
                              self.group_rate,
                              n_layer,
                              self.order_prompt,
                              self.rms_norm,
                              self.drop_out_in_block,
                              self.drop_path)
            )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head = nn.Sequential(
            nn.Conv1d(self.trans_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, self.cls_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, xyz):
        features = self.emb(xyz)
        xyz = xyz[..., 0:3]
        # encoding
        xyz_and_feats = []
        for i in range(len(self.enc_layers)):
            center, group_input = self.down_sampling[i](xyz, features) if i != 0 else (xyz, features)
            xyz, features = self.order_merging_enc[i](center, group_input)
            xyz_and_feats.append((xyz, features))

        # decoding
        xyz = xyz_and_feats[-1][0]
        features = self.fc(xyz_and_feats[-1][1])
        for i in range(len(self.dec_layers)):
            center, group_input = self.up_sampling[i](xyz, features, xyz_and_feats[- i - 1][0], xyz_and_feats[- i - 1][1])
            xyz, features = self.order_merging_dec[i](center, group_input)

        logit = self.cls_head(self.norm(features).transpose(2, 1))
        return logit.transpose(2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    def to_categorical(y, num_classes):
        return torch.eye(num_classes)[y.numpy(),]


    cat_prompt = to_categorical(torch.tensor([1, 2, 3, 4]), 16)

    inputs = torch.rand((4, 1024, 3), )
    cfg = EasyDict(
        {
            "NAME": Cloud_Mamba_Part,
            "input_dim": 3,
            "trans_dim": 36,
            "group_rate": 3,
            "encoder_layers": [1, 3, 2, 1],
            "decoder_layers": [1, 1, 1, 1],
            "order_prompt": True,
            "rms_norm": False,
            "drop_path": 0.3,
            "group_size": 16,
            "num_group": 1024,
            "cls_dim": 50,
            "category": 16
        }
    )

    model = Cloud_Mamba_Part(cfg).cuda()
    outputs = model(inputs.cuda(), cat_prompt.cuda())
    print(outputs)
