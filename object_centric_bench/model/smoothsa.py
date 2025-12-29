"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
from types import MethodType

from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class SmoothSA(nn.Module):
    """
    Slot Attention with Re-Initialization and Self-Distillation.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,  # trunc_bp=false: bad
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        __class__.reset_parameters(  # reset self.decode: no difference
            [self.encode_posit_embed, self.encode_project, self.aggregat]
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
        feature = self.encode_backbone(input).detach()  # (b,c,h,w)
        b, c, h, w = feature.shape

        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        qinit, query = self.initializ(encode, condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        clue = rearrange(feature, "b c h w -> b (h w) c")
        recon, attent2 = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)

        return feature, qinit, slotz, attent, attent2, recon


class SmoothSAVideo(SmoothSA):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,  # trunc_bp=false: bad
        transit,
        decode,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            decode,
        )
        self.transit = transit
        __class__.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c0, h0, w0 = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz = None
        attent = []

        for i in range(t):
            if i == 0:
                qinit0, query_i = self.initializ(
                    encode[:, 0, :, :], None if condit is None else condit[:, 0, :, :]
                )  # (b,n,c)
            else:
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attent_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            slotz = (  # (b,i+1,n,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attent.append(attent_i)  # t*(b,n,h*w)

        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, qinit0, slotz, attent, attent2, recon


class NormalSharedPreheated(nn.Module):  #  > normal separate

    def __init__(self, num, emb_dim, kv_dim):
        super().__init__()
        self.num = num
        self.emb_dim = emb_dim
        self.kv_dim = kv_dim

        # zero mean > xavier_uniform, xavier_normal or randn mean
        self.mean = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))
        self.logstd = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        self.qproj_kv = nn.Linear(kv_dim, emb_dim)  # > ln, fc, lnfc, fcln, mlp
        self.qinit = nn.TransformerDecoderLayer(  # > SwappedTransformerDecoderLayer, i.e., different qdim and kvdim
            emb_dim,
            # kv_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=0,  # 0 vs 0.1, 0.5: good for arifg
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.qinit.forward = MethodType(forward_switch_sa_ca, self.qinit)
        if self.qinit.norm_first:
            del self.qinit.norm2  # good for arifg
            self.qinit.norm2 = lambda _: _

        self.logstd2 = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        """
        ###### <<<--- shared enc_proj, built-in tfdb as preheat

        detach first 0.1                                        ############################################
        ---
        smoothsa_r-coco,0.2896±0.0079,0.4200±0.0024,0.3339±0.0036,0.3192±0.0038
        smoothsav_r-ytvis,0.4390±0.0118,0.6180±0.0302,0.4107±0.0083,0.4053±0.0084

        detach first 0.1 qproj_kv=lnfc
        ---
        smoothsa_r-coco,0.2754±0.0022,0.4121±0.0040,0.3281±0.0006,0.3136±0.0006
        smoothsav_r-ytvis,0.4661±0.0152,0.6469±0.0376,0.4212±0.0089,0.4151±0.0096

        detach first 0.1 qproj_kv=fcln
        ---
        smoothsa_r-coco,0.2762±0.0039,0.4112±0.0023,0.3291±0.0022,0.3143±0.0021
        smoothsav_r-ytvis,0.4745±0.0152,0.6565±0.0277,0.4308±0.0097,0.4244±0.0095

        detach first 0.2
        ---
        smoothsa_r-coco,0.2896±0.0008,0.4056±0.0038,0.3282±0.0007,0.3122±0.0008
        smoothsav_r-ytvis,0.4324±0.0108,0.6062±0.0276,0.4062±0.0110,0.4024±0.0107

        detach first 0.5
        ---
        smoothsa_r-coco,0.2695±0.0020,0.4046±0.0015,0.3181±0.0018,0.3021±0.0019
        smoothsav_r-ytvis,0.4127±0.0108,0.5979±0.0152,0.3996±0.0095,0.3968±0.0097

        ### --->>>

        ###### <<<--- shared enc_proj, reimpl (qdim != kvdim) tfdb as preheat

        shared encode_project + reimpl swappedtfdb + ln
        ---
        smoothsa_r-coco,0.2738±0.0017,0.4150±0.0030,0.3285±0.0007,0.3139±0.0008
        smoothsav_r-ytvis,0.4916±0.0093,0.6247±0.0341,0.4450±0.0078,0.4387±0.0061

        shared encode_project + reimpl swappedtfdb + fc         ############################################
        ---
        smoothsa_r-coco,0.2892±0.0042,0.4126±0.0026,0.3339±0.0025,0.3196±0.0025
        smoothsav_r-ytvis,0.4445±0.0106,0.6251±0.0353,0.4154±0.0108,0.4101±0.0104

        shared encode_project + reimpl swappedtfdb + fcln
        ---
        smoothsa_r-coco,0.2745±0.0016,0.4127±0.0003,0.3280±0.0011,0.3130±0.0011
        smoothsav_r-ytvis,0.4794±0.0107,0.6226±0.0410,0.4335±0.0126,0.4274±0.0120

        shared encode_project + reimpl swappedtfdb + lnfc       ############################################
        ---
        smoothsa_r-coco,0.2730±0.0026,0.4159±0.0062,0.3273±0.0013,0.3125±0.0014
        smoothsav_r-ytvis,0.4710±0.0050,0.6510±0.0091,0.4303±0.0033,0.4241±0.0033

        shared encode_project + reimlp swappedtfdb + lnmlpln
        ---
        smoothsa_r-coco,0.2814±0.0069,0.4096±0.0069,0.3281±0.0015,0.3132±0.0016
        smoothsav_r-ytvis,0.4643±0.0033,0.6246±0.0221,0.4238±0.0013,0.4183±0.0014

        ### --->>>

        ###### <<<--- separate enc_proj | mlp

        separate encode_project | mlp + reimpl swapped tfdb
        ---
        smoothsa_r-coco,0.2614±0.0027,0.3966±0.0014,0.3131±0.0003,0.2973±0.0003
        smoothsav_r-ytvis,0.4134±0.0045,0.5955±0.0052,0.4049±0.0051,0.4024±0.0052

        separate encode_project | fc + reimpl swapped tfdb
        ---
        smoothsa_r-coco,0.2670±0.0020,0.3917±0.0029,0.3154±0.0003,0.2991±0.0001
        smoothsav_r-ytvis,0.4115±0.0055,0.5963±0.0123,0.4003±0.0048,0.3976±0.0049

        ### --->>>
        """
        self.register_buffer("detach_flag", pt.tensor(1, dtype=pt.bool))

    def forward(self, encode, n: int = None):
        b, hw, c = encode.shape
        self_num = self.num
        if n is not None:
            self_num = n

        mean = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(mean)  # better than not
        smpl = mean + randn * self.logstd.exp()

        if self.detach_flag:  # detach initial > always detach
            encode = encode.detach()
        qinit = self.qinit(smpl, self.qproj_kv(encode))
        # in training, start from smpl as qinit than switch to real qinit: bad

        if self.training:
            randn2 = pt.randn_like(qinit)  # better than not
            query = qinit.detach() + randn2 * self.logstd2.exp()
        else:
            query = qinit.detach()
        # align qinit with slotz > align qinit+std with slotz
        return qinit, query  # > query, query.detach()


from .basic import MLP


class NormalMlpPreheated(nn.Module):

    def __init__(self, in_dim, dims, kv_dim):
        super().__init__()
        emb_dim = dims[-1]
        self.emb_dim = emb_dim
        self.kv_dim = kv_dim

        self.mlp = MLP(in_dim, dims, "post", 0)
        self.logstd = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))
        """
        ln0post + randn0/1*1      bbox_pad_value=-1     bbox_pad_size=auto    ############################################
        ---
        smoothsav_c-movi_c,0.5104±0.0127,0.6938±0.0030,0.3198±0.0057,0.3058±0.0058

        ln0post + randn0/1*1      bbox_pad_value=-1     bbox_pad_size=auto      randn0/1*0.1 @bbox_padding
        ---
        smoothsav_c-movi_c,0.5181±0.0281,0.6827±0.0012,0.3087±0.0088,0.2947±0.0095

        ln0post ................................        bbox_pad_size=max
        ---
        similar to ``ln0post``

        ln0post ..........................................................  randn0/1*0 @bbox_padding
        ---
        very good for ari and arifg but very bad for mbo and miou

        ===

        ln0post (+ rand0*1)
        ---
        smoothsav_c-movi_c,0.4224±0.0133,0.6873±0.0060,0.3110±0.0060,0.2997±0.0061

        ln0post + rand0*0.1
        ---
        smoothsav_c-movi_c,0.4042±0.0280,0.7041±0.0114,0.3021±0.0054,0.2900±0.0058

        ln0post + rand0*0.01
        ---
        smoothsav_c-movi_c,0.3785±0.0342,0.7015±0.0132,0.2882±0.0081,0.2751±0.0089

        ln0post + rand0*0
        ---
        smoothsav_c-movi_c,0.5497±0.0055,0.6716±0.0260,0.2511±0.0021,0.2261±0.0018

        no ln0 (just mlp)
        ---
        smoothsav_c-movi_c,0.4097±0.0020,0.6859±0.0072,0.3065±0.0010,0.2950±0.0011

        ln0post + rand0_slotshared
        ---
        smoothsav_c-movi_c,0.5519±0.0116,0.6565±0.0166,0.2533±0.0061,0.2272±0.0064

        ln0post + mlpdo0.1 (no rand0)
        ---
        smoothsav_c-movi_c,0.5439±0.0100,0.6682±0.0355,0.2493±0.0061,0.2238±0.0055

        ===

        ln0post + randn0    qproj_kv=fc
        ---
        smoothsav_c-movi_c,0.4195±0.0058,0.6852±0.0092,0.3117±0.0048,0.3006±0.0049

        ln0post + randn0    qproj_kv=lnfc
        ---
        smoothsav_c-movi_c,0.4107±0.0219,0.6627±0.0109,0.2896±0.0076,0.2755±0.0076

        ln0post + randn0    qproj_kv=ln
        ---
        smoothsav_c-movi_c,0.3962±0.0190,0.6650±0.0034,0.2881±0.0084,0.2736±0.0086

        ln0post + randn0    qproj_kv=fc
        ---
        smoothsav_c-movi_c,0.4076±0.0111,0.6803±0.0114,0.3034±0.0018,0.2912±0.0024

        ln0post + randn0    qproj_kv=lnfc
        ---
        smoothsav_c-movi_c,0.37         ,0.65         ,0.28,        ,0.26
        """

        self.qproj_kv = nn.Linear(kv_dim, emb_dim)
        self.qinit = nn.TransformerDecoderLayer(  # SwappedTransformerDecoderLayer
            emb_dim,
            # kv_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=0,  # 0 vs 0.1, 0.5: good for arifg
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.qinit.forward = MethodType(forward_switch_sa_ca, self.qinit)
        if self.qinit.norm_first:
            del self.qinit.norm2  # good for arifg
            self.qinit.norm2 = lambda _: _

        self.logstd2 = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        self.register_buffer("detach_flag", pt.tensor(1, dtype=pt.bool))

    def forward(self, encode, condit):
        """
        - encode: shape=(b,h*w,c)
        - condit: shape=(b,n,c)
        """
        mean = self.mlp(condit)
        randn = pt.randn_like(mean)  # better than not
        smpl = mean + randn * self.logstd.exp()  # > share_randn0/1_on_pad (different on batch)

        if self.detach_flag:
            encode = encode.detach()
        qinit = self.qinit(smpl, self.qproj_kv(encode))

        if self.training:
            randn2 = pt.randn_like(qinit)  # better than not
            query = qinit.detach() + randn2 * self.logstd2.exp()
        else:
            query = qinit.detach()
        return qinit, query  # > query, query.detach()


def forward_switch_sa_ca(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    tgt_is_causal: bool = False,
    memory_is_causal: bool = False,
):
    x = tgt
    if self.norm_first:
        x = x + self._mha_block(  # swape self-att and cross-att
            self.norm2(x),
            memory,
            memory_mask,
            memory_key_padding_mask,
            memory_is_causal,
        )
        x = x + self._sa_block(
            self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
        )
        x = x + self._ff_block(self.norm3(x))
    else:
        x = self.norm2(
            x
            + self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
        )
        x = self.norm1(
            x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        )
        x = self.norm3(x + self._ff_block(x))

    return x


class SwappedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    reimpltfdb (this module) + qproj_kv=lnfc    只对YTVIS有益
    ---
    smoothsa_r-clevrtex,0.7569±0.0011,0.8181±0.0022,0.5996±0.0031,0.5828±0.0036
    best val of smoothsa_r-clevrtex: 59.12 [28 21]
    smoothsa_r-coco,0.2768±0.0012,0.4126±0.0007,0.3287±0.0006,0.3142±0.0006
    best val of smoothsa_r-coco: 32.14 [17 26]
    smoothsa_r-voc,0.3382±0.0030,0.3445±0.0002,0.4484±0.0006,0.4363±0.0003
    best val of smoothsa_r-voc: 44.230000000000004 [27 37]
    smoothsav_c-movi_c,0.4628±0.0058,0.6761±0.0005,0.2853±0.0014,0.2680±0.0026
    best val of smoothsav_c-movi_c: 27.66 [36 31]
    smoothsav_c-movi_d,0.3986±0.0060,0.7267±0.0062,0.2947±0.0037,0.2807±0.0040
    best val of smoothsav_c-movi_d: 28.77 [29 31]
    smoothsav_r-ytvis,0.4839±0.0043,0.6124±0.0234,0.4346±0.0071,0.4294±0.0071
    best val of smoothsav_r-ytvis: 43.2 [25 35]

    reimpltfdb + qproj_kv=fc                    只对YTVIS有益
    ---
    smoothsa_r-clevrtex,0.7673±0.0034,0.8098±0.0034,0.5996±0.0030,0.5825±0.0036
    best val of smoothsa_r-clevrtex: 59.099999999999994 [19 21]
    smoothsa_r-coco,0.2897±0.0046,0.4069±0.0076,0.3340±0.0006,0.3194±0.0007
    best val of smoothsa_r-coco: 32.67 [18 23]
    smoothsa_r-voc,0.3483±0.0008,0.3475±0.0005,0.4570±0.0013,0.4447±0.0015
    best val of smoothsa_r-voc: 45.09 [31 39]
    smoothsav_c-movi_c,0.4748±0.0024,0.6970±0.0069,0.3038±0.0016,0.2896±0.0017
    best val of smoothsav_c-movi_c: 29.67 [36 35]
    smoothsav_c-movi_d,0.4388±0.0013,0.7114±0.0006,0.3117±0.0004,0.2989±0.0002
    best val of smoothsav_c-movi_d: 30.53 [29 29]
    smoothsav_r-ytvis,0.4466±0.0079,0.6336±0.0304,0.4168±0.0103,0.4110±0.0113
    best val of smoothsav_r-ytvis: 41.39 [33 37]
    """

    def __init__(
        self,
        d_model,
        kv_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=0.00001,
        batch_first=False,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        if self.norm_first is True:
            self.norm2 = nn.Identity()  # actually 1st norm after swapping
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout,
            bias,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=batch_first,
        )

    forward = forward_switch_sa_ca
