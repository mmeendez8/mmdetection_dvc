import copy
import time

import fire
import yaml
from mmcv import Config
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def merge_configs(dataset, model, schedule=None, runtime=None):
    """Merge different configs in a single file and updates it with DVC params.

    Args:
        dataset: Path to dataset file
        model: Path to model file
        schedule: Path to schedule file
        runtime: Path to runtime file

    """
    base_config = dict()

    for config_base_file in [dataset, model, schedule, runtime]:
        if config_base_file is not None:
            base_config.update(Config._file2dict(config_base_file)[0])

    base_config = update_config_with_dvc_params(base_config)

    config = Config(base_config)

    return config


def update_config_with_dvc_params(base_config):
    """Updates base_config dictionary with DVC params.

    Args:
        base_config: Mmdetection config in dict format

    """
    params = yaml.safe_load(open("params.yaml"))

    if params is None:
        return base_config

    def _update(config, params):
        for key, value in params.items():
            if isinstance(value, dict):
                config[key] = _update(config.get(key, {}), value)
            else:
                config[key] = value
        return config

    return _update(base_config, params)


def train(dataset, model, schedule, runtime, work_dir="./experiments", gpu_id=0):
    """Train mmdetection model. Simplified from https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py

    Args:
        dataset: Path to dataset file
        model: Path to model file
        schedule: Path to schedule file
        runtime: Path to runtime file
        work_dir: Output directory for checkpoints and log files
        gpu_ids: Id of gpu to use

    """
    cfg = merge_configs(dataset, model, schedule, runtime)

    cfg = update_config_with_dvc_params(cfg)

    cfg.gpu_ids = [gpu_id]
    cfg.work_dir = work_dir
    cfg.seed = None

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    model = build_detector(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=None,
    )


if __name__ == "__main__":
    fire.Fire(train)
