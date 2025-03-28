import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
import fiftyone.utils.random as four


"""
This is the file to filter out COCO data,  only people and traffic light
"""
def export_yolo_data(
        samples,
        export_dir,
        classes,
        label_field="ground_truth",
        split=None
):
    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(
                samples,
                export_dir,
                classes,
                label_field,
                split
            )
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )


dataset = fo.load_dataset("coco-2017-train")
# dataset.delete()
# detection_model = YOLO("yolov8n.pt")


dataset = foz.load_zoo_dataset("coco-2017",
                               split="train",
                               classes=["person","traffic light"])

dataset.filter_labels("ground_truth", (F("label") == "person") | (F("label") == "traffic light")).save()
dataset.untag_samples("train")
dataset.save()
# model=YOLO("yolov8n.pt")
# model.to("cuda")

# dataset.apply_model(model,label_field="person")
session = fo.launch_app(dataset)

EXPORT_DIR = "/trained_model"
classes = ["person", "traffic light"]

four.random_split(
    dataset,
    {"train": 0.8, "val": 0.2}
)
export_yolo_data(
    dataset,
    EXPORT_DIR,
    classes,
    split=["train", "val"]
)
