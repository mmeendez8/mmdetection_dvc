from typing import Optional

import fire
import torch
import torch.utils.data as data_utils
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from train import merge_configs


def eval(
    dataset,
    model,
    checkpoint_file,
    output_dir,
    n_samples: Optional[int] = None,
    score_threshold: Optional[float] = 0.3,
) -> None:
    """Eval mmdetection model. Simplified from https://github.com/open-mmlab/mmdetection/blob/master/tools/test.py

    Args:
        dataset: Path to dataset file
        model: Path to model file
        checkpoint_file: Path to checkpoint file
        output_dir: Output directory for images
        score_threshold: Minimum score for a detection to be considered

    """
    cfg = merge_configs(dataset, model)

    cfg.data.val.test_mode = True

    dataset = build_dataset(cfg.data.val)

    if n_samples is not None:
        dataset = data_utils.Subset(dataset, torch.arange(n_samples))

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_detector(cfg=cfg.model, test_cfg=cfg.get("test_cfg"))

    checkpoint = load_checkpoint(model, checkpoint_file, map_location="cpu")

    model.CLASSES = checkpoint["meta"]["CLASSES"]
    model = MMDataParallel(model, device_ids=[0])

    single_gpu_test(
        model=model,
        data_loader=data_loader,
        show=False,
        out_dir=output_dir,
        show_score_thr=score_threshold,
    )


if __name__ == "__main__":
    fire.Fire(eval)
