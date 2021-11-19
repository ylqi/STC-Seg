# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import os
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.structures import Boxes, BoxMode

# from .coco import load_coco_json, load_sem_seg

"""
This file contains functions to register a COCO-format dataset to the DatasetCatalog.
"""

__all__ = ["register_kitti_instances", "get_kitti_instances_meta"]

# cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
#                 'Tram', 'Misc', 'DontCare']
cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
                'Tram', 'Misc']

# KITTI_CATEGORIES = [
#     {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": cats[0]},
#     {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": cats[1]},
#     {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": cats[2]},
#     {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": cats[3]},
#     {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": cats[4]},
#     {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": cats[5]},
#     {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": cats[6]},
#     {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": cats[7]},
#     {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": cats[8]},
# ]
KITTI_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": cats[0]},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": cats[1]},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": cats[2]},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": cats[3]},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": cats[4]},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": cats[5]},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": cats[6]},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": cats[7]},
]

def load_kitti_json(json_path, img_path, name):
    with open(json_path) as f:
        for i, line in enumerate(f):
            images = json.loads(line) # run through each line
            #img = json.dumps(images, indent = 2) #for a better representation

    dataset_dicts = []
    
    for i, image in enumerate(images['images']):
        record = {}
        # print("registering image")
        
        filename = os.path.join(img_path, image["file_name"])
        _id = image["id"]

        #if the file doesn't exist
        # height, width = cv2.imread(filename).shape[:2]
        try:
            height, width = cv2.imread(filename).shape[:2]
        except AttributeError:
            print("File doesn't exist")
            continue

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = _id #

        objs = [] #many instances in 1 record
        
        for anno in images['annotations']:
            anno_id = anno["image_id"]
            category_id = anno["category_id"]
            #check if the image id from image data same with annotation and include only person and person sitting
            #although in the JSON file it is number 8, the annotation start from 1
            if anno_id == _id: 
                # area = anno["area"] 
                instance_id = anno["id"]
                # print("Iter {2} Instance {1} In image {0}".format(filename, instance_id,i)) 
                px = anno["bbox"][0]
                py = anno["bbox"][1]
                p_width = anno["bbox"][2]
                p_height = anno["bbox"][3]
            
                obj = {"bbox": [px,py,p_width,p_height],
                        "bbox_mode": BoxMode.XYWH_ABS, #it's not XYXY but XYWH
                        # "area": area,
                        "segmentation":[],
                        
                        "category_id": category_id - 1, #set things only classes person
                        "iscrowd": 0}
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

        # if i == 200: #200 iterations
        #   break

    return dataset_dicts


def register_kitti_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_kitti_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def get_kitti_instances_meta(dataset_name):
    thing_ids = [k["id"] for k in KITTI_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in KITTI_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 8 or len(thing_ids) == 9, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in KITTI_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret