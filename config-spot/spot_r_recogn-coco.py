import importlib
import sys

import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    MSCOCO,
    ClPadToMax1,
    DefaultCollate,
    Xywh2Ltrb,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    CrossEntropyLoss,
    L1Loss,
    ClassAccuracy,
    TensorSize,
    BoxIoU,
    CbLinearCosine,
    Callback,
    HandleLog,
    SaveModel,
)
from object_centric_bench.model import (
    MLP,
    ObjDiscovRecogn,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

max_num = 7
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384
ncls = 90 + 1
cbox = 4

total_step = 10000
val_interval = total_step // 40
batch_size_t = 64 // 2  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 1e-3

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
    extra_keys=["segment", "bbox", "clazz"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=MSCOCO,
    data_file="coco/val.lmdb",
    extra_keys=["segment", "bbox", "clazz"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(  # (b,h,w,s) (b,s,c) (b,s)
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment", "bbox", "clazz"], dims=[2, 0, 0]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

sys.path.append(".")
cfg_dict = importlib.import_module("spot_r-coco").__dict__
discov = cfg_dict["model"]

recogn = dict(
    type=MLP, in_dim=emb_dim, dims=[emb_dim * 2, ncls + cbox], ln=None, dropout=0.1
)

model = dict(
    type=ObjDiscovRecogn,
    discov=discov,
    recogn=recogn,
    slotz_idx=1,
    attpd_idx=2,  # TODO XXX 3
    segpd_func=lambda _: ptnf.one_hot(
        interpolat_argmax_attent(_.detach(), size=resolut0).long()
    ).bool(),
    slotz_rearr="b s c -> b s c",
    segpd_rearr="b h w s -> b (h w) s",
    seggt_rearr="b h w s -> b (h w) s",
    clsgt_rearr="b s -> b s",
    boxgt_rearr="b s c -> b s c",
    ncls=ncls,
    cbox=cbox,
    thresh_iou=1e-3,
)
model_imap = dict(
    input="batch.image",
    # condit=None
    seggt="batch.segment",
    clsgt="batch.clazz",
    boxgt="batch.bbox",
)
model_omap = ["slotz", "clspd", "boxpd", "clsgt", "boxgt", "rcidx"]
ckpt_map = [  # target<-source
    ["m.discov.", "m."],  # load trained OCL weights into discov
]
freez = [r"^m\.discov\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = None

loss_fn = dict(
    ce=dict(
        metric=dict(type=CrossEntropyLoss),
        map=dict(input="output.clspd", target="output.clsgt"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    l1=dict(
        metric=dict(type=L1Loss),
        map=dict(input="output.boxpd", target="output.boxgt"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
        weight=1,
    ),
)
acc_fn_t = dict(
    top1=dict(
        metric=dict(type=ClassAccuracy, topk=1),
        map=dict(input="output.clspd", target="output.clsgt"),  # (?,ncls)->(?,)
    ),
    top3=dict(
        metric=dict(type=ClassAccuracy, topk=3),
        map=dict(input="output.clspd", target="output.clsgt"),  # (?,ncls)->(?,)
    ),
    iou=dict(
        metric=dict(type=BoxIoU),
        map=dict(input="output.boxpd", target="output.boxgt"),  # (?,cbox)
        transforms=dict(type=Xywh2Ltrb, keys=["input"]),
    ),
    num=dict(  # count the number of matched objects
        metric=dict(type=TensorSize, dim=0),
        map=dict(input="output.clsgt"),
    ),
)
acc_fn_v = acc_fn_t.copy()

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.image", "batch.segment", "batch.bbox", "batch.clazz"]],
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
]
callback_t = [
    dict(type=Callback, before_step=before_step),
    dict(
        type=HandleLog,
        log_file=...,
        ikeys=[["loss.ce", "loss.l1", "acc.top1", "acc.top3", "acc.iou"], ["acc.num"]],
        okeys=[["ce", "l1", "top1", "top3", "iou"], ["num"]],
        ops=["mean", "sum"],
    ),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1]),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
