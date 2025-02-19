# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path

from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed, Sampler

from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from .dataset import YOLODataset
from .utils import PIN_MEMORY


class IncrementalDataLoader:
    def __init__(self, ir_loader, fusion_loader, task_scheduler):
        self.ir_loader = iter(ir_loader)
        self.fusion_loader = iter(fusion_loader)
        self.task_scheduler = task_scheduler

    def __iter__(self):
        # 生成批次序列 (例: 40融合 + 60红外 = 100批次)
        batch_sequence = ['fusion'] * self.task_scheduler['fusion_batches'] + ['ir'] * self.task_scheduler['ir_batches']
        random.shuffle(batch_sequence)  # 关键步骤：打乱顺序避免模式震荡

        for batch_type in batch_sequence:
            if batch_type == 'fusion':
                try:
                    yield next(self.fusion_loader), True
                except StopIteration:
                    self.fusion_loader = iter(self.fusion_loader)
                    yield next(self.fusion_loader), True
            else:
                try:
                    yield next(self.ir_loader), False
                except StopIteration:
                    self.ir_loader = iter(self.ir_loader)
                    yield next(self.ir_loader), False


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that supports resetting iterator while maintaining epoch boundaries.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_length = len(self.batch_sampler)  # 获取一个epoch的批次数
        self._iterator = None

    def __iter__(self):
        """Generates exactly one epoch of data, then stops."""
        self._iterator = super().__iter__()  # 创建普通迭代器
        for _ in range(self.epoch_length):   # 仅生成固定数量的批次
            yield next(self._iterator)

    def reset(self):
        """Reset the iterator manually."""
        self._iterator = super().__iter__()  # 重新创建迭代器


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


class MixedBatchSampler:
    def __init__(self, ir_len, irvis_len, ir_prob, batch_size, batches_per_epoch):
        self.ir_indices = np.arange(ir_len)
        self.irvis_indices = np.arange(irvis_len)
        self.ir_prob = ir_prob
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.ir_length = ir_len
        self.irvis_length = irvis_len

    def __iter__(self):
        np.random.shuffle(self.ir_indices)
        np.random.shuffle(self.irvis_indices)

        num_ir_batches = int(self.batches_per_epoch * self.ir_prob)
        num_irvis_batches = self.batches_per_epoch - num_ir_batches
        batch_types = ['ir'] * num_ir_batches + ['irvis'] * num_irvis_batches
        np.random.shuffle(batch_types)

        ir_idx, irvis_idx = 0, 0

        for bt in batch_types:
            if bt == 'ir':
                if ir_idx + self.batch_size > self.ir_length:
                    ir_idx = 0
                    np.random.shuffle(self.ir_indices)
                batch = self.ir_indices[ir_idx:ir_idx+self.batch_size]
                ir_idx += self.batch_size
                yield [('ir', idx) for idx in batch]
            else:
                if irvis_idx + self.batch_size > self.irvis_length:
                    irvis_idx = 0
                    np.random.shuffle(self.irvis_indices)
                batch = self.irvis_indices[irvis_idx:irvis_idx+self.batch_size]
                irvis_idx += self.batch_size
                yield [('irvis', idx) for idx in batch]

    def __len__(self):
        return self.batches_per_epoch


class CustomBatchSampler(Sampler):
    def __init__(self, imfiles, batch_size=16, shuffle=True, drop_last=False, fusion_mode=False):
        super().__init__(None)
        self.imfiles = imfiles
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.folder_to_indices = self._group_indices()  # 仅分组，不打乱

    def _group_indices(self):
        """按文件夹分组索引（不预先打乱）"""
        folder_to_indices = defaultdict(list)
        for idx, path in enumerate(self.imfiles):
            folder = os.path.dirname(path)
            folder_to_indices[folder].append(idx)
        return folder_to_indices

    def __iter__(self):
        # 每个 epoch 动态生成批次
        folders = list(self.folder_to_indices.keys())
        if self.shuffle:
            random.shuffle(folders)

        all_indices = []
        for folder in folders:
            indices = self.folder_to_indices[folder]
            if self.shuffle:
                random.shuffle(indices)
            all_indices.extend(indices)

        # 生成当前 epoch 的所有批次
        batches = [
            all_indices[i:i+self.batch_size]
            for i in range(0, len(all_indices), self.batch_size)
        ]
        if self.drop_last and len(batches[-1]) != self.batch_size:
            batches = batches[:-1]

        return iter(batches)

    def __len__(self):
        """准确计算批次数"""
        total = sum(len(indices) for indices in self.folder_to_indices.values())
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0
    )


class StageAwareDataLoader:
    """支持分阶段动态混合的DataLoader容器"""

    def __init__(self,
                 dataset,
                 batch_size: int = 32,
                 base_seed: int = 6148914691236517205,
                 num_workers: int = 4):
        self.device_count = torch.cuda.device_count()
        self.dataset = dataset
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.num_workers = min(os.cpu_count() // max(self.device_count, 1), num_workers)

        # 计算各阶段参数
        self.loader = None
        self.get_loader(1)

    def _create_loader(self, ir_ratio: float) -> dataloader.DataLoader:
        """动态创建混合比例DataLoader"""
        batches_per_epoch = len(self.dataset.ir_dataset) // self.batch_size

        sampler = MixedBatchSampler(
            ir_len=len(self.dataset.ir_dataset),
            irvis_len=len(self.dataset.irvis_dataset),
            ir_prob=ir_ratio,
            batch_size=self.batch_size,
            batches_per_epoch=batches_per_epoch
        )

        self.loader = dataloader.DataLoader(
            self.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            num_workers=self.num_workers,
            worker_init_fn=seed_worker
        )

    def get_loader(self, ir_ratio) -> dataloader.DataLoader:
        """根据当前epoch返回对应DataLoader"""
        if ir_ratio == 1:
            sampler = CustomBatchSampler(self.dataset.ir_dataset.im_files, self.batch_size, shuffle=True)
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205 + RANK)
            self.loader = InfiniteDataLoader(
                    dataset=self.dataset.ir_dataset,
                    # shuffle=shuffle and sampler is None,
                    num_workers=self.num_workers,
                    sampler=None,
                    batch_sampler=sampler,
                    # batch_size=1,
                    pin_memory=PIN_MEMORY,
                    collate_fn=getattr(self.dataset.ir_dataset, "collate_fn", None),
                    worker_init_fn=seed_worker,
                    generator=generator,
                )
        else:
            self._create_loader(ir_ratio=ir_ratio)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    sampler = CustomBatchSampler(dataset.im_files, batch, shuffle=shuffle) if rank == -1 else (
       distributed.DistributedSampler(dataset, shuffle=shuffle))

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        # shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=None,
        batch_sampler=sampler,
        # batch_size=1,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
