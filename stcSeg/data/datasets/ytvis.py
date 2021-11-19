import json
import os
import numpy as np
import cv2

from pycocotools import mask as maskUtils
from pycocotools import coco

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


cats = ['person','giant_panda','lizard','parrot','skateboard','sedan',
      'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
      'train','horse','turtle','bear','motorbike','giraffe','leopard',
      'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
      'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
      'tennis_racket']



YTVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1},
    {"color": [119, 11, 32], "isthing": 1, "id": 2},
    {"color": [0, 0, 142], "isthing": 1, "id": 3},
    {"color": [0, 0, 230], "isthing": 1, "id": 4},
    {"color": [106, 0, 228], "isthing": 1, "id": 5},
    {"color": [0, 60, 100], "isthing": 1, "id": 6},
    {"color": [0, 80, 100], "isthing": 1, "id": 7},
    {"color": [0, 0, 70], "isthing": 1, "id": 8},
    {"color": [0, 0, 192], "isthing": 1, "id": 9},
    {"color": [250, 170, 30], "isthing": 1, "id": 10},
    {"color": [100, 170, 30], "isthing": 1, "id": 11},
    {"color": [220, 220, 0], "isthing": 1, "id": 12},
    {"color": [175, 116, 175], "isthing": 1, "id": 13},
    {"color": [250, 0, 30], "isthing": 1, "id": 14},
    {"color": [165, 42, 42], "isthing": 1, "id": 15},
    {"color": [255, 77, 255], "isthing": 1, "id": 16},
    {"color": [0, 226, 252], "isthing": 1, "id": 17},
    {"color": [182, 182, 255], "isthing": 1, "id": 18},
    {"color": [0, 82, 0], "isthing": 1, "id": 19},
    {"color": [120, 166, 157], "isthing": 1, "id": 20},
    {"color": [110, 76, 0], "isthing": 1, "id": 21},
    {"color": [174, 57, 255], "isthing": 1, "id": 22},
    {"color": [199, 100, 0], "isthing": 1, "id": 23},
    {"color": [72, 0, 118], "isthing": 1, "id": 24},
    {"color": [255, 179, 240], "isthing": 1, "id": 25},
    {"color": [0, 125, 92], "isthing": 1, "id": 26},
    {"color": [209, 0, 151], "isthing": 1, "id": 27},
    {"color": [188, 208, 182], "isthing": 1, "id": 28},
    {"color": [0, 220, 176], "isthing": 1, "id": 29},
    {"color": [255, 99, 164], "isthing": 1, "id": 30},
    {"color": [92, 0, 73], "isthing": 1, "id": 31},
    {"color": [133, 129, 255], "isthing": 1, "id": 32},
    {"color": [78, 180, 255], "isthing": 1, "id": 33},
    {"color": [0, 228, 0], "isthing": 1, "id": 34},
    {"color": [174, 255, 243], "isthing": 1, "id": 35},
    {"color": [45, 89, 255], "isthing": 1, "id": 36},
    {"color": [134, 134, 103], "isthing": 1, "id": 37},
    {"color": [145, 148, 174], "isthing": 1, "id": 38},
    {"color": [255, 208, 186], "isthing": 1, "id": 39},
    {"color": [197, 226, 255], "isthing": 1, "id": 40},
]

def get_ytvis_instances_meta(dataset_name):
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == len(cats), len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [cats[k["id"] - 1] for k in YTVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):

    f = json.load(open(json_file))  
    ytvis_annotations = []
    image_sizes = []
    for v_id, video in enumerate(f['videos']):

        img_width = video['width']
        img_height = video['height']
        image_sizes.append({"w": img_width, "h": img_height})


    annot_dict = {}

    if 'annotations' in f:
        video_ids = []
        for item in f['annotations']:
            track_id = item['id']
            video_id = item['video_id'] - 1

            for frame_id, bbox in enumerate(item['bboxes']):
                if bbox is None:
                    continue

                w = image_sizes[video_id]['w']
                h = image_sizes[video_id]['h']

                segm = item['segmentations'][frame_id]
                segm = segmToRLE(segm, (h, w))
                # mask to poly
                mask = np.ascontiguousarray(coco.maskUtils.decode(segm))
                _, contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
                poly = []

                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:
                        poly.append(contour)
                if len(poly) == 0:
                    continue

                annot = {
                    "category_id": item['category_id'] - 1,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    "segmentation": poly

                }

                
                file_name = f['videos'][video_id]['file_names'][frame_id]

                if file_name not in annot_dict.keys():
                    annot_dict[file_name] = {"f_objs": [annot], "video_id": video_id}
                else:
                    annot_dict[file_name]["f_objs"].append(annot)

    ytvis_annots = []
    for i, file_name in enumerate(annot_dict):

        annots = annot_dict[file_name]

        f_objs = annots["f_objs"]
        w = image_sizes[annots["video_id"]]['w']
        h = image_sizes[annots["video_id"]]['h']

        frame_data = {
            "file_name": os.path.join(image_root, file_name),
            "image_id": i,
            "height": h,
            "width": w,
            "annotations": f_objs
        }
        ytvis_annots.append(frame_data)

    # print('total image #: {}'.format(image_cnt))
    # json.dump(out, open(out_path + '{}.json'.format(split), 'w'))

    return ytvis_annots


def register_ytvis_instances(name, metadata, json_file, image_root):
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
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, evaluator_type="ytvis", **metadata
    )