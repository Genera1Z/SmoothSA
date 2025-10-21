from copy import deepcopy
import importlib
import sys

from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    MSCOCO,
    PadToMax1,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    AttentMatchLoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveModel,
)
from object_centric_bench.model import (
    SPOTDistill,
    NormalSeparat,
)
from object_centric_bench.util import Compose
from object_centric_bench.util_model import interpolat_argmax_attent

### global

max_num = 7
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384

total_step = 100000  # 100000 better
val_interval = total_step // 40
batch_size_t = 64 // 2  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 4e-4 / 2  # scale with batch_size

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["image", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["image", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type=CenterCrop, keys=["image", "segment"], size=None),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type=MSCOCO,
    data_file="coco/train.lmdb",
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=MSCOCO,
    data_file="coco/val.lmdb",
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(type=PadToMax1, keys=["segment"], dims=[2])
collate_fn_v = collate_fn_t

### model

sys.path.append(".")
cfg_dict = importlib.import_module("spot_r-coco").__dict__
spot_teacher = cfg_dict["model"]

spot_student = deepcopy(spot_teacher)
spot_student["initializ"]["type"] = NormalSeparat
spot_student["aggregat"]["trunc_bp"] = "bi-level"
spot_student["finetune"] = True

model = dict(
    type=SPOTDistill,
    teacher=spot_teacher,
    student=spot_student,
)
model_imap = dict(input="batch.image")
model_omap = ["feature", "slotz", "attent", "attent2", "recon", "attent2_t"]
ckpt_map = [  # target<-source
    ["m.teacher.", "m."],
    ["m.student.encode_backbone.", "m.encode_backbone."],
    ["m.student.encode_posit_embed.", "m.encode_posit_embed."],
    ["m.student.encode_project.", "m.encode_project."],
    ["m.student.aggregat.", "m.aggregat."],
    ["m.student.decode.", "m.decode."],
]
freez = [
    r"^m\.teacher\..*",
    r"^m\.student\.encode_backbone\.1\.(cls_token|reg_token|pos_embed|patch_embed|blocks\.[0-7]\.).*",
    r"^m\.student\.encode_backbone2\..*",
]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=0.3, norm_type="inf")

loss_fn = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    match=dict(
        metric=dict(type=AttentMatchLoss),
        map=dict(input="output.attent", target="output.attent2_t"),
        weight=0.005,
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment2", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b h w s -> b (h w) s"),
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
        type=Lambda, ikeys=[["batch.image", "batch.segment"]], func=lambda _: _.cuda()
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,  # 1e3 > 1e4
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attent2"]],  # (b,s,h,w) -> (b,h,w,s)
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment2"]],
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
