import json
from pathlib import Path

import fire
from PIL import Image


def prepare_data(coco_file: str, output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(coco_file, "r") as f:
        coco = json.load(f)

    # Reset coco categories numbering ids
    new_category_id = {}
    for i, category in enumerate(coco["categories"], start=1):
        new_category_id[category["id"]] = i
        category["id"] = i

    # Add ids, area and new category numbering to annotations
    for i, ann in enumerate(coco["annotations"]):
        ann["id"] = i
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        ann["category_id"] = new_category_id[ann["category_id"]]
        ann["iscrowd"] = 0

    # Add image width and height
    for img in coco["images"]:
        file_name = Path("data/coco_sample/train_sample") / img["file_name"]
        img["width"], img["height"] = Image.open(file_name).size

    with open(output_file, "w") as f:
        json.dump(coco, f, indent=2)


if __name__ == "__main__":
    fire.Fire(prepare_data)
