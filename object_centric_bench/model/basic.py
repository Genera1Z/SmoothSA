import re

from einops import rearrange
import timm
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from ..util import DictTool


####


class ModelWrap(nn.Module):  # TODO XXX TensorDictModule

    def __init__(self, m: nn.Module, imap, omap):
        """
        - imap: dict or list.
            If keys in batch mismatches with keys in model.forward, use dict, ie, {key_in_batch: key_in_forward};
            If not, use list.
        - omap: list
        """
        super().__init__()
        assert isinstance(imap, (dict, list, tuple))
        assert isinstance(omap, (list, tuple))
        self.m = m
        self.imap = imap if isinstance(imap, dict) else {_: _ for _ in imap}
        self.omap = omap

    # def forward(self, input: dict) -> dict:
    def forward(self, **pack: dict) -> dict:
        # input2 = {k: input[v] for k, v in self.imap.items()}
        input2 = {k: DictTool.getattr(pack, v) for k, v in self.imap.items()}
        output = self.m(**input2)
        if not isinstance(output, (list, tuple)):
            output = [output]
        assert len(self.omap) == len(output)
        output2 = dict(zip(self.omap, output))
        return output2

    def load(self, ckpt_file: str, ckpt_map: list, verbose=True):
        state_dict = pt.load(ckpt_file, map_location="cpu", weights_only=True)
        if ckpt_map is None:
            if verbose:
                print("fully")
            self.load_state_dict(state_dict)  # TODO XXX , False
        elif isinstance(ckpt_map, (list, tuple)):
            for dst, src in ckpt_map:
                dkeys = [_ for _ in self.state_dict() if _.startswith(dst)]
                skeys = [_ for _ in state_dict if _.startswith(src)]
                assert len(dkeys) == len(skeys)  # > 0
                if len(dkeys) == 0:
                    print(
                        f"[{__class__.__name__}.load WARNING] ``{dst}, {src}`` has no matched keys !!!"
                    )
                for dk, sk in zip(dkeys, skeys):
                    if verbose:
                        print(dk, sk)
                    self.state_dict()[dk].data[...] = state_dict[sk]
        else:
            raise "ValueError"
        if verbose:
            print(f"checkpoint ``{ckpt_file}`` loaded")

    def save(self, save_file, weights_only=True, key=r".*"):
        if weights_only:
            save_obj = self.state_dict()
            save_obj = {k: v for k, v in save_obj.items() if re.match(key, k)}
        else:
            save_obj = self
        pt.save(save_obj, save_file)

    def freez(self, freez: list, verbose=True):
        for n, p in self.named_parameters():
            for f in freez:
                if bool(re.match(f, n)):
                    p.requires_grad = False
        if verbose:
            [print(k, v.requires_grad) for k, v in self.named_parameters()]

    def group_params(self, coarse=r"^.*", fine=dict()):
        """Group model parameters by coarse and fine filters.

        - coarse: coarse filter; regex string
        - fine: fine filter for grouping and adding extras; {regex1: dict(lr_mult=0.5, wd_mult=0),..}
        """
        # coarse filtering
        named_params = dict(self.named_parameters())
        named_params = {
            k: v for k, v in named_params.items() if bool(re.match(coarse, k))
        }
        if not fine:
            params = []
            for k, v in named_params.items():
                if v.requires_grad:
                    print(f"{k} - to train, require grad")
                    params.append(v)
                else:
                    print(f"{k} - skipped, not require grad")
            return params

        # fine filtering
        param_groups = {k: dict(params=[]) for k in fine}  # TODO lr
        names = list(named_params.keys())
        for n, p in named_params.items():
            for g, (k, v) in enumerate(fine.items()):
                assert isinstance(v, dict)
                if bool(re.match(k, n)):
                    cursor = names.pop(0)
                    assert cursor == n  # ensure no missing or overlap
                    if p.requires_grad:
                        print(f"{n} - #{g}, {v}")
                        param_groups[k]["params"].append(p)
                        param_groups[k].update(v)
                    else:
                        print(f"{n} - #{g}, skipped, not require grad")

        param_groups = {k: v for k, v in param_groups.items() if len(v["params"])}
        return list(param_groups.values())


class Sequential(nn.Sequential):
    """"""

    def __init__(self, modules: list):
        super().__init__(*modules)

    def forward(self, input):
        for module in self:
            if isinstance(input, (list, tuple)):  # TODO control in init
                input = module(*input)
            else:
                input = module(input)
        return input


ModuleList = nn.ModuleList


####


Embedding = nn.Embedding


Conv2d = nn.Conv2d


PixelShuffle = nn.PixelShuffle


ConvTranspose2d = nn.ConvTranspose2d


AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d


Identity = nn.Identity


ReLU = nn.ReLU


GELU = nn.GELU


SiLU = nn.SiLU


Mish = nn.Mish


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, interp="bilinear"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.interp = interp

    def forward(self, input):
        return ptnf.interpolate(input, self.size, self.scale_factor, self.interp)


Dropout = nn.Dropout


Linear = nn.Linear


GroupNorm = nn.GroupNorm


LayerNorm = nn.LayerNorm


####


MultiheadAttention = nn.MultiheadAttention


TransformerEncoderLayer = nn.TransformerEncoderLayer


TransformerDecoderLayer = nn.TransformerDecoderLayer


TransformerEncoder = nn.TransformerEncoder


TransformerDecoder = nn.TransformerDecoder


###


class MLP(nn.Sequential):
    """"""

    def __init__(self, in_dim, dims, ln: str = None, dropout=0):
        """
        - ln: None for no layernorm, 'pre' for pre-norm, 'post' for post-norm
        """
        assert ln in [None, "pre", "post"]

        num = len(dims)
        layers = []
        ci = in_dim

        if ln == "pre":
            layers.append(nn.LayerNorm(ci))

        for i, c in enumerate(dims):
            if i + 1 < num:
                block = [
                    nn.Linear(ci, c),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout else None,
                ]
            else:
                block = [nn.Linear(ci, c)]

            layers.extend([_ for _ in block if _])
            ci = c

        if ln == "post":
            layers.append(nn.LayerNorm(ci))

        super().__init__(*layers)


class DINO2ViT(nn.Module):
    """
    https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8
    """

    def __init__(
        self,
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        in_size=518,
        rearrange=True,
        norm_out=True,
    ):
        super().__init__()
        # dict(
        #     patch_size=14,
        #     embed_dim=384,
        #     depth=12,
        #     num_heads=6,
        #     init_values=1e-05,
        #     reg_tokens=4,
        #     no_embed_class=True,
        #     pretrained_cfg="lvd142m",
        #     pretrained_cfg_overlay=None,
        #     cache_dir=None,
        # )
        model = timm.create_model(model_name, pretrained=True, img_size=in_size)
        self.in_size = model.patch_embed.img_size[0]
        assert self.in_size == in_size
        self.patch_size = model.patch_embed.patch_size[0]
        assert self.patch_size == 14

        self.cls_token = model.cls_token
        self.reg_token = model.reg_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = model.norm if norm_out else nn.Identity()

        for k, v in model.__dict__.items():
            if any(
                [
                    k.startswith("__") and k.endswith("__"),
                    k.startswith("_"),
                    isinstance(v, nn.Module),
                    isinstance(v, nn.Parameter),
                    hasattr(self, k),
                ]
            ):
                print(f"[{__class__.__name__}] skip {k}")
                continue
            else:
                print(f"[{__class__.__name__}] copy {k}")
                setattr(self, k, v)
        assert hasattr(self, "num_prefix_tokens")

        __class__._pos_embed = model.__class__._pos_embed
        __class__.forward_features = model.__class__.forward_features

        self.rearrange = rearrange
        self.out_size = in_size // self.patch_size
        assert self.out_size <= 518 // 14

    def forward(self, input):
        """
        input: shape=(b,c,h,w), float
        """
        # with pt.inference_mode(True):  # infer+compile: errors
        feature = self.forward_features(input)
        if self.rearrange:
            feature = feature[:, self.num_prefix_tokens :, :]  # remove class token
            feature = rearrange(feature, "b (h w) c -> b c h w", h=self.out_size)
        return feature  # .clone()
