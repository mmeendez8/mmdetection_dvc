# dataset settings
dataset_type = "CocoDataset"
data_root = "processed_data/"
img_folder = "data/coco_sample/train_sample"

CLASSES = ["chair", "couch", "tv", "remote", "book", "vase"]

img_scale = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train_split_with_ann_id.json",
        img_prefix=img_folder,
        pipeline=train_pipeline,
        classes=CLASSES,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_split_with_ann_id.json",
        img_prefix=img_folder,
        pipeline=test_pipeline,
        classes=CLASSES,
    ),
)

evaluation = dict(interval=1, metric="bbox")
