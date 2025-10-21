from copy import deepcopy

from einops import rearrange
import numpy as np
import torch as pt
import torch.nn as nn


class SPOT(nn.Module):
    """
    SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        finetune=False,  # finetune self.encode_backbone
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_backbone2 = deepcopy(encode_backbone) if finetune else None
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.decode]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: image, shape=(b,c,h,w)
        - condit: condition, shape=(b,n,c)
        """
        feature = self.encode_backbone(input)  # (b,c,h,w)
        if self.encode_backbone2:  # finetune self.encode_backbone
            with pt.inference_mode():
                feature2 = self.encode_backbone2(input).detach()  # (b,c,h,w)
            feature2 = feature2.clone()
        else:
            feature2 = feature.detach().clone()
        b, c, h, w = feature.shape

        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        clue = rearrange(feature2, "b c h w -> b (h w) c")
        recon, attent2 = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)

        return feature2, slotz, attent, attent2, recon
        # segment acc: attent ~= attent2 ???


class SPOTDistill(nn.Module):

    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, input, condit=None):
        with pt.inference_mode(True):
            _, _, _, attent2_t, _ = self.teacher(input, condit)
        feature, slotz, attent, attent2, recon = self.student(input, condit)
        return feature, slotz, attent, attent2, recon, attent2_t

    def train(self, mode=True):
        self = super().train(mode)
        self.teacher.eval()
        return self


class AR9TransformerDecoder(nn.Module):
    """SPOT's decoder. Auto-regressive Transformer decoder with 9 permutations."""

    def __init__(
        self,
        resolut,
        vfm_dim,
        posit_embed,  # 1d
        project1,
        project2,
        backbone,
        readout,
        perm_t="random",
        perm_v="default",  # ``all`` is not as beneficial as claimed
    ):
        super().__init__()

        self.perm_t = perm_t
        self.perm_v = perm_v
        perm = pt.as_tensor(__class__.generate_permutations(*resolut))
        self.register_buffer("perm", perm, persistent=False)
        if self.perm_t == "default":
            self.perm_v = "default"
        self.perm_idx = list(range(len(self.perm)))

        self.bos = nn.Parameter(pt.randn(len(self.perm), 1, 1, vfm_dim) * vfm_dim**-0.5)
        self.posit_embed = posit_embed
        self.register_buffer(
            "mask",
            pt.triu(pt.ones([np.prod(resolut)] * 2, dtype=pt.bool), 1),
            persistent=False,
        )
        self.project1 = project1
        self.project2 = project2

        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1  # very beneficial
        backbone.layers[0].norm1 = nn.Identity()  # very beneficial
        self.backbone = backbone
        self.readout = readout

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True  # obtain the attention weights

        def attent_hook_forward(module, args, output):
            self._attent = output[1]

        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(
            attent_hook_forward
        )

    @staticmethod
    def generate_permutations(h, w):
        perm_default = np.arange(h * w)
        perm_default_2d = perm_default.reshape(h, w)

        hs = tuple(range(h))
        ws = tuple(range(w))
        perm_topleft = [perm_default_2d[r, c] for c in ws for r in hs]
        perm_topright = [perm_default_2d[r, c] for c in ws[::-1] for r in hs]
        perm_righttop = [perm_default_2d[r, c] for r in hs for c in ws[::-1]]
        perm_bottomright = [perm_default_2d[r, c] for c in ws[::-1] for r in hs[::-1]]
        perm_rightbottom = [perm_default_2d[r, c] for r in hs[::-1] for c in ws[::-1]]
        perm_bottomleft = [perm_default_2d[r, c] for c in ws for r in hs[::-1]]
        perm_leftbottom = [perm_default_2d[r, c] for r in hs[::-1] for c in ws]

        perm_spiral = []
        A = np.rot90(perm_default_2d.copy(), k=1)
        while A.size:
            perm_spiral.append(A[0])  # take first row
            A = A[1:].T[::-1]  # cut off first row and rotate counterclockwise
        perm_spiral = np.concatenate(perm_spiral)[::-1]

        return (
            perm_default.tolist(),
            perm_topleft,
            perm_topright,
            perm_righttop,
            perm_bottomright,
            perm_rightbottom,
            perm_bottomleft,
            perm_leftbottom,
            perm_spiral.tolist(),
        )

    def forward(self, input, slotz):
        """
        - input: shape=(b,m=h*w,c)
        - slotz: shape=(b,n,c)
        """
        if self.training:
            if self.perm_t == "default":
                which = [0]
            elif self.perm_t == "random":
                which = [np.random.choice(self.perm_idx)]
            elif self.perm_t == "all":
                which = self.perm_idx
            else:
                raise ValueError
        else:
            if self.perm_v == "default":
                which = [0]
            elif self.perm_v == "random":
                which = [np.random.choice(self.perm_idx)]
            elif self.perm_v == "all":
                which = self.perm_idx
            else:
                raise ValueError

        output = []
        attent = []

        for perm_idx in which:
            perm_i = self.perm[perm_idx]
            inv_perm_i = perm_i.argsort()

            bos_i = self.bos[perm_idx].expand(input.shape[0], -1, -1)
            query_i = pt.cat([bos_i, input[:, perm_i, :][:, :-1, :]], dim=1)
            query_i = self.project1(query_i)
            memory_i = self.project2(slotz)
            output_i = self.backbone(self.norm0(query_i), memory_i, tgt_mask=self.mask)

            attent_i = self._attent

            output_i = output_i[:, inv_perm_i, :]
            attent_i = attent_i[:, inv_perm_i, :]

            attent.append(attent_i)
            output.append(output_i)

        output = pt.stack(output).mean(0)  # (b,m,c)
        attent = pt.stack(attent).mean(0).permute(0, 2, 1)  # (b,n,m)
        return output, attent
