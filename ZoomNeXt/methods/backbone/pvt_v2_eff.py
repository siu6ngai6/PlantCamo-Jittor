from functools import partial
import jittor as jt
from jittor import nn
import math


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,linear=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.gauss_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                jt.init.zero_(m.bias)

    def execute(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> jt.Var:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = jt.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = jt.ones(L, S, dtype=jt.bool).tril(diagonal=0)
        attn_bias[jt.logical_not(temp_mask)] = float("-inf")
        # attn_bias.to(query.dtype)
        attn_bias = jt.array(attn_bias, query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == jt.bool:
            attn_bias[jt.logical_not(temp_mask)] = float("-inf")
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = nn.softmax(attn_weight, dim=-1)
    attn_weight = nn.dropout(attn_weight, dropout_p, is_train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.gauss_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                jt.init.zero_(m.bias)

    def execute(self, x, H, W): 
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)  # built-in scale

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            linear=linear,
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.gauss_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                jt.init.zero_(m.bias)

    def execute(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

import collections.abc
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.gauss_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                jt.init.zero_(m.bias)

    def execute(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        linear=linear,
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.gauss_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                jt.init.zero_(m.bias)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def no_weight_decay(self):
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}

    def extract_endpoints(self, x):
        B = x.shape[0]
        endpoints = dict()
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                if self.use_checkpoint:
                    x = jt.checkpoint(blk, x, H, W)
                else:
                    x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # print(i + 2, x.shape)
            endpoints["reduction_{}".format(i + 2)] = x
        return endpoints

    def execute(self, x):
        endpoints = self.extract_endpoints(x)
        return endpoints


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def execute(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

cfg = {'url': '', 'num_classes': 1000, 'input_size': (3, 384, 384), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'fixed_input_size': True, 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'first_conv': 'patch_embed.proj', 'classifier': 'head', 'license': 'apache-2.0'}

def pvt_v2_eff_b0(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, 
        embed_dims=[32, 64, 160, 256], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-06), 
        depths=[2, 2, 2, 2], 
        sr_ratios=[8, 4, 2, 1], 
        **kwargs
    )
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model

def pvt_v2_eff_b1(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model

def pvt_v2_eff_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model

def pvt_v2_eff_b3(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model

def pvt_v2_eff_b4(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model

def pvt_v2_eff_b5(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    )
    model.default_cfg = cfg
    if pretrained:
        url = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth'
        state_dict = jt.load(url)
        
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        model.load_parameters(state_dict)
    return model
