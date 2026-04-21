"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from pathlib import Path
import pickle as pkl

from einops import rearrange
from tqdm import tqdm
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util import concurrent_pool
from ..util_datum import mask_segment_to_bbox_np, draw_segmentation_np


class MOVi(ptud.Dataset):
    """Multi-Object Video (MOVi) datasets.
    - https://github.com/google-research/kubric/tree/main/challenges/movi
    - https://console.cloud.google.com/storage/browser/kubric-public/tfds

    Frame size in a scene:
    - timestep=24, height=256, width=256, channel=3.

    Example
    ```
    dataset = MOVi(
        data_file="movi_c/train.lmdb",
        extra_keys=["segment", "bbox", "flow", "depth"],
        base_dir=Path("/media/GeneralZ/Storage/Static/datasets"),
    )
    for sample in dataset:
        dataset.visualiz(
            video=sample["video"].permute(0, 2, 3, 1).numpy(),
            segment=sample["segment"].numpy(),
            bbox=sample["bbox"].numpy(),
            flow=sample["flow"].permute(0, 2, 3, 1).numpy(),
            depth=sample["depth"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "bbox", "flow", "depth"],
        transform0=lambda **_: _,  # for t-slice only
        transform=lambda **_: _,
        base_dir: Path = None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        env.close()

        self.extra_keys = extra_keys
        self.transform0 = transform0
        self.transform = transform

    def __getitem__(self, index):
        """
        - video: (t=24,c=3,h,w), uint8 | float32
        - segment: (t,h,w,s), uint8 -> bool
        - bbox: (t,s,c=4), float32. both side normalized ltrb, only foreground
        - flow: (t,c=3,h,w), uint8 -> float32
        - depth: (t,h,w), float32
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample0 = self.transform0(**sample0)  # clip videos in advance for efficiency
        sample1 = {}

        video0 = np.array(  # rgb
            [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in sample0["video"]]
        )
        video = pt.from_numpy(video0).permute(0, 3, 1, 2)
        sample1["video"] = video  # (t,c,h,w) uint8

        if "segment" in self.extra_keys:
            segment0 = np.array(
                [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in sample0["segment"]]
            )
            segment = pt.from_numpy(segment0)
            sample1["segment"] = segment  # (t,h,w) uint8
            s0 = 1 + sample0["s"]  # bg+fg

        if "flow" in self.extra_keys:
            flowd = sample0["flow"]
            flowd["data"] = np.stack(
                [
                    np.array([cv2.imdecode(__, cv2.IMREAD_UNCHANGED) for __ in _])
                    for _ in flowd["data"]
                ],
                -1,
            )
            flow0 = __class__.unpack_uint16_to_float32(**flowd)
            flow0 = __class__.flow_to_rgb(flow0)
            flow = pt.from_numpy(flow0).permute(0, 3, 1, 2)
            sample1["flow"] = flow  # (t,c=3,h,w) uint8

        if "depth" in self.extra_keys:
            depthd = sample0["depth"]
            depthd["data"] = np.array(
                [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in depthd["data"]]
            )
            depth0 = __class__.unpack_uint16_to_float32(**depthd)
            depth = pt.from_numpy(depth0)
            sample1["depth"] = depth  # (t,h,w) float32

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (t,h,w); index format
            # (t,h,w,s); mask format
            segment2_ = ptnf.one_hot(segment2.long(), s0).bool()

            t, h, w, _ = segment2_.shape

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment2_.any([0, 1, 2])  # (s,)
            segment3 = segment2_[:, :, :, cond]
            sample2["segment"] = segment3  # (t,h,w,s) bool

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(  # skip bg
                    segment3[:, :, :, 1 if cond[0] else 0 :], "t h w s -> h w (t s)"
                )
                bbox2_ = pt.from_numpy(  # (t*s,c=4)
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w  # normalize
                bbox2[:, :, 1::2] /= h
                sample2["bbox"] = bbox2  # (t,s,c=4) float32

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        src_dir="/media/GeneralZ/Storage/Static/datasets_raw",
        tfds_name="movi_c/256x256:1.0.0",
        dst_dir=Path("movi_c"),
    ):
        """
        Convert the original TFRecord files into one LMDB file, saving 10x storage space.

        Note: This requires the following TensorFlow-series libs, which could mess up your environment,
        so to run this part you had better just install them soley in a separate environment.
        ```
        clu==0.0.10
        tensorflow_cpu
        tensorflow_datasets
        ```

        Download MOVi series datasets. Remember to install gsutil first https://cloud.google.com/storage/docs/gsutil_install
        ```bash
        cd local/path/to/movi_c/256x256/
        gsutil -m cp -r gs://kubric-public/tfds/movi_c/256x256/1.0.0 .
        # download movi_a, b, d, e, f in the similar way if needed
        ```
        Then the file structure is like:
        - movi_c/256x256/1.0.0  # !!! make sure !!!
            - movi_c-test.tfrecord-*****-of****
            - movi_c-train.tfrecord-*****-of****
            - movi_c-validation.tfrecord-*****-of****

        Finally create a Python script with the following content at the project root, and execute it:
        ```python
        from object_centric_bench.datum import MOVi
        MOVi.convert_dataset()  # remember to change default paths to yours
        ```
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        splits = dict(train="train", val="validation")

        from clu import deterministic_data
        import tensorflow as tf
        import tensorflow.python.framework.ops as tfpfo
        import tensorflow_datasets as tfds

        _gpus = tf.config.list_physical_devices("GPU")
        [tf.config.experimental.set_memory_growth(_, True) for _ in _gpus]

        for split, split_name in splits.items():
            print(split)

            dataset_builder = tfds.builder(tfds_name, data_dir=src_dir)
            dataset_split = deterministic_data.get_read_instruction_for_host(
                split_name,
                dataset_builder.info.splits[split_name].num_examples,
            )
            dataset = deterministic_data.create_dataset(
                dataset_builder,
                split=dataset_split,
                batch_dims=(),
                num_epochs=1,
                shuffle=False,
            )

            dst_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(dst_file)

            keys = []
            txn = lmdb_env.begin(write=True)

            for i, sample0 in enumerate(tqdm(dataset)):
                sample1 = __class__.tensorflow2pytorch_nested_mapping(
                    sample0, tfpfo, tf
                )
                sample2 = __class__.video_from_tfds(sample1)
                sample2 = __class__.bbox_sparse_to_dense(sample2)

                video = sample2["video"]  # (t,h,w,c) uint8
                segment = sample2["segment"]  # (t,h,w) uint8
                bbox = sample2["bbox"]  # (t,n,c=4) float32, both side normalized
                bbox = bbox[:, :, [1, 0, 3, 2]]
                flow = sample2["flow"]  # (t,h,w,c=3) uint8 dict
                depth = sample2["depth"]  # (t,h,w) uint32 dict

                s = segment.max()  # only fg
                assert s == bbox.shape[1]

                # segment_msk = ptnf.one_hot(pt.from_numpy(segment).long()).bool().numpy()
                # flow0 = __class__.unpack_uint16_to_float32(**flow)
                # flow0 = __class__.flow_to_rgb(flow0)
                # depth0 = __class__.unpack_uint16_to_float32(**depth)
                # __class__.visualiz(video, segment_msk, bbox, flow0, depth0, wait=0)

                assert video.shape == (24, 256, 256, 3) and video.dtype == np.uint8
                assert segment.shape == (24, 256, 256) and segment.dtype == np.uint8
                assert (
                    bbox.ndim == 3
                    and bbox.shape[0] == 24
                    and bbox.shape[2] == 4
                    and bbox.dtype == np.float32
                )
                assert (
                    flow["data"].shape == (24, 256, 256, 2)
                    and flow["data"].dtype == np.uint16
                )
                assert (
                    depth["data"].shape == (24, 256, 256)
                    and depth["data"].dtype == np.uint16
                )

                sample_key = f"{i:06d}".encode("ascii")
                keys.append(sample_key)

                # For compression rate, cv2's png (image) is as good as PyAV's libx264rgb or ffv1 (video).
                enc_param = [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    9,
                    cv2.IMWRITE_PNG_STRATEGY,
                    cv2.IMWRITE_PNG_STRATEGY_FILTERED,
                ]
                efunc = lambda _: cv2.imencode(".png", _, enc_param)[1]
                sample_dict = dict(  # lossless video encoding always has some losses
                    video=concurrent_pool(efunc, [video]),
                    segment=concurrent_pool(efunc, [segment]),
                    s=s,
                    bbox=bbox,
                    flow=dict(
                        data=[
                            concurrent_pool(efunc, [flow["data"][:, :, :, _]])
                            for _ in range(flow["data"].shape[-1])
                        ],
                        min=flow["min"],
                        max=flow["max"],
                    ),
                    depth=dict(
                        data=concurrent_pool(efunc, [depth["data"]]),
                        min=depth["min"],
                        max=depth["max"],
                    ),
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (i + 1) % 64 == 0:  # write_freq
                    print(f"{i + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

            txn.commit()
            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

    @staticmethod
    def tensorflow2pytorch_nested_mapping(mapping: dict, tfpfo, tf):
        mapping2 = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                value2 = __class__.tensorflow2pytorch_nested_mapping(value, tfpfo, tf)
            elif isinstance(value, tfpfo.EagerTensor):
                value2 = value.numpy()
            elif isinstance(value, tf.RaggedTensor):
                value2 = value.to_list()
            else:
                raise "NotImplemented"
            mapping2[key] = value2
        return mapping2

    @staticmethod
    def video_from_tfds(pack: dict) -> dict:
        """Adopted from SAVi official implementation VideoFromTfds class."""
        video = pack["video"].astype("uint8")  # (t,h,w,c)

        track = pack["instances"]["bbox_frames"]
        bbox = pack["instances"]["bboxes"]

        flow_range = pack["metadata"]["backward_flow_range"]
        flow = pack["backward_flow"]  # 0~65535 (t,h,w,c=2)

        depth_range = pack["metadata"]["depth_range"]
        depth = pack["depth"][:, :, :, 0]  # 0~65535 (t,h,w)

        segment = pack["segmentations"][:, :, :, 0]  # (t,h,w)

        return dict(
            video=video,  # uint8 (24,256,256,3)
            segment=segment,  # uint8 (24,256,256)
            bbox=dict(track=track, bbox=bbox),  # float32
            flow=dict(  # uint16 (24,256,256,2)
                data=flow, min=flow_range[0], max=flow_range[1]
            ),
            depth=dict(  # uint16 (24,256,256)
                data=depth, min=depth_range[0], max=depth_range[1]
            ),
        )

    @staticmethod
    def bbox_sparse_to_dense(pack: dict, notrack=0) -> dict:  # TODO notrack=-1
        """Adopted from SAVi official implementation SparseToDenseAnnotation class."""

        def densify_bbox(tracks: list, bboxs_s: list, timestep: int):
            assert len(tracks) == len(bboxs_s)

            null_box = np.array([notrack] * 4, dtype="float32")
            bboxs_d = np.tile(null_box, [timestep, len(tracks), 1])  # (t,n,c=4)

            for i, (track, bbox_s) in enumerate(zip(tracks, bboxs_s)):
                idx = np.array(track, dtype="int64")
                value = np.array(bbox_s, dtype="float32")
                bboxs_d[idx, i] = value

            return bboxs_d  # (t,n+1,c=4)

        track = pack["bbox"]["track"]
        bbox0 = pack["bbox"]["bbox"]

        segment = pack["segment"]
        assert segment.max() <= len(track)

        bbox = densify_bbox(track, bbox0, segment.shape[0])
        pack["bbox"] = bbox

        return pack

    @staticmethod
    def unpack_uint16_to_float32(data, min, max):
        assert data.dtype == np.uint16
        return data.astype("float32") / 65535.0 * (max - min) + min

    @staticmethod
    def flow_to_rgb(flow, flow_scale=50.0, hsv_scale=[180.0, 255.0, 255.0]):
        # ``torchvision.utils.flow_to_image`` got strange result
        assert flow.ndim == 4
        hypot = lambda a, b: (a**2.0 + b**2.0) ** 0.5  # sqrt(a^2 + b^2)

        flow_scale = flow_scale / hypot(*flow.shape[2:4])
        hsv_scale = np.array(hsv_scale, dtype="float32")[None, None, None]

        x, y = flow[..., 0], flow[..., 1]

        h = np.arctan2(y, x)  # motion angle
        h = (h / np.pi + 1.0) / 2.0
        s = hypot(y, x)  # motion magnitude
        s = np.clip(s * flow_scale, 0.0, 1.0)
        v = np.ones_like(h)

        hsv = np.stack([h, s, v], axis=3)
        hsv = (hsv * hsv_scale).astype("uint8")
        rgb = np.array([cv2.cvtColor(_, cv2.COLOR_HSV2RGB) for _ in hsv])

        return rgb

    @staticmethod
    def visualiz(video, segment=None, bbox=None, flow=None, depth=None, wait=0):
        """
        - video: (t,h,w,c=3) uint8, rgb format
        - segment: (t,h,w,s) bool, mask format
        - bbox: (t,s,c=4) float32, ltrb format, dual normalized
        - flow: (t,h,w,c=3) uint8, rgb format
        - depth: (t,h,w) float32
        """
        t, h, w, cv = video.shape
        assert cv == 3 and video.dtype == np.uint8

        if segment is not None:
            t, h, w, cs = segment.shape
            assert segment.dtype == bool

        if bbox is not None:
            t, s, cb = bbox.shape
            assert cb == 4 and bbox.dtype == np.float32
            if segment is not None:
                assert cs - s in [0, 1]
            bbox = (bbox.copy() * [w, h, w, h]).round().astype(int)

        if flow is not None:
            t, h, w, cf = flow.shape
            assert cf == 3 and flow.dtype == np.uint8

        if depth is not None:
            t, h, w = depth.shape
            assert depth.dtype == np.float32
            dmin = depth.min()
            dmax = depth.max()
            depth = (depth - dmin) / (dmax - dmin)

        c1 = (255, 255, 255)
        imgs = []
        segs = []

        for ti, img in enumerate(video):
            cv2.imshow("v", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            imgs.append(img)

            if segment is not None:
                seg = draw_segmentation_np(img, segment[ti, :, :, :], alpha=0.75)

                if bbox is not None and bbox.shape[0]:
                    for box in bbox[ti, :, :]:
                        seg = cv2.rectangle(seg, box[:2], box[2:], color=c1)

                cv2.imshow("s", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
                segs.append(seg)

            if flow is not None:
                cv2.imshow("f", cv2.cvtColor(flow[ti, :, :, :], cv2.COLOR_RGB2BGR))

            if depth is not None:
                cv2.imshow("d", depth[ti, :, :])

            cv2.waitKey(wait)

        return imgs, segs
