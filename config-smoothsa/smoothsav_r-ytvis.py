from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    StridedRandomSlice1,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    YTVIS,
    ClPadToMax1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    SlotContrastLoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    CbSquarewave,
    Callback,
    AverageLog,
    SaveModel,
)
from object_centric_bench.model import (
    SmoothSAVideo,
    Sequential,
    Interpolate,
    DINO2ViT,
    Identity,
    MLP,
    NormalSharedPreheated,
    SlotAttention,
    RSFQTransit,
    ARRandTransformerDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

max_num = 6 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384

total_step = 50000  # 100000 better
val_interval = total_step // 40
batch_size_t = 32 // 4  # 64 better
batch_size_v = 1
num_work = 4
lr = 2e-4 / 4  # scale with batch_size

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # (t=20,c,h,w) (t,n,c=4) (t,h,w)
    dict(type=StridedRandomSlice1, keys=["video", "segment"], dim=0, size=5),
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=YTVIS,
    data_file="ytvis/train.lmdb",
    ts=20,
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=YTVIS,
    data_file="ytvis/val.lmdb",
    ts=None,
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

model = dict(
    type=SmoothSAVideo,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, scale_factor=0.875, interp="bicubic"),
            dict(
                type=DINO2ViT,
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP, in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(
        type=NormalSharedPreheated, num=max_num, emb_dim=emb_dim, kv_dim=vfm_dim
    ),
    aggregat=dict(
        type=SlotAttention,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp="bi-level",
    ),
    transit=dict(
        type=RSFQTransit,
        dt=5,  # XXX XXX XXX
        ci=vfm_dim,
        c=emb_dim,
        nhead=4,
        expanz=4,
        pdo=0.5,  # XXX XXX XXX
        norm_first=False,
        bias=False,
    ),
    decode=dict(
        type=ARRandTransformerDecoder,
        vfm_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(  # fc>fc+ln
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(  # fc+ln>fc
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=4,  # 4@384, 6@768
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type=Identity),
    ),
)
model_imap = dict(input="batch.video")
model_omap = ["feature", "qinit", "slotz", "attenta", "recon", "attentd"]
ckpt_map = []  # target<-source
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=0.05)

loss_fn_t = loss_fn_v = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    ssc=dict(
        metric=dict(type=SlotContrastLoss),
        map=dict(input="output.slotz"),
        weight=0.5,
    ),
    preheat=dict(
        metric=dict(type=MSELoss),  # (b,s,c) (b,t,s,c)
        map=dict(input="output.qinit", target="output.slotz"),
        transform=dict(
            type=Lambda, ikeys=[["target"]], func=lambda _: _[:, 0, :, :].detach()
        ),
        weight=100,
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
    dict(
        type=CbSquarewave,
        assigns=["model.m.initializ.detach_flag.data[...]=value"],
        points=[0, total_step // 10, total_step],
        values=[1, 0],
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attentd"]],  # (b,t,s,h,w) -> (b,t,h,w,s)
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
