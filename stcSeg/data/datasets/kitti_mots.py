import os
import cv2
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco
import numpy as np
import re

# from prettytable import PrettyTable
# table = PrettyTable(['Video Name', 'Frame Num', 'Annotation Num'])

classes_correspondence = {
    'Car': 0,
    'Pedestrian': 1,
}

# mots cat_id --> coco cat_id # esto no tira
coco_correspondence = {
    0: 2,
    1: 0,
}

KITTI_MOTS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Car"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Pedestrian"}]


def get_kiti_mots_dicts(name, images_folder, annots_folder, is_train, train_percentage=0.75, image_extension="png"):
    assert os.path.exists(images_folder), images_folder
    assert os.path.exists(annots_folder)

    # annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))

    # n_train_seqences = int(len(annot_files) * train_percentage)
    # train_sequences = annot_files[:n_train_seqences]
    # test_sequences = annot_files[n_train_seqences:]

    # sequences = train_sequences if is_train else test_sequences

    split = "train" if is_train else "val"

    annot_files = []
    seqmap_file = open("datasets/kitti_mots/splits/%s.seqmap" % (split), 'r')
    for line in seqmap_file:
        fields = line.split(" ")
        annot_files.append(os.path.join(annots_folder, "%04d.txt" % int(fields[0])))

    # --------------------------- If is Full Train -------------------------
    if is_train and re.search("full", name):
        annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))
    # --------------------------- If is Full Train -------------------------

    # -------------------------- On specific video -------------------------
    # annot_files = sorted(glob(os.path.join(annots_folder, "0018.txt")))
    # -------------------------- On specific video -------------------------

    sequences = annot_files

    kitti_mots_annotations = []
    
    for seq_file in sequences:
        seq_images_path = os.path.join(images_folder, seq_file.split("/")[-1].split(".")[0])
        kitti_mots_annotations += mots_annots_to_coco(seq_images_path, seq_file, image_extension)
    # print(table)

    return kitti_mots_annotations


def mots_annots_to_coco(images_path, txt_file, image_extension):
    assert os.path.exists(txt_file)
    n_seq = int(txt_file.split("/")[-1].split(".")[0])


    mots_annots = []
    with open(txt_file, 'r') as f:
        annots = f.readlines()
        annots = [l.split() for l in annots]

        annots = np.array(annots)
        # print(np.unique(annots[:, 0].astype('uint16')))

        # table.add_row(["%04d" % n_seq, "%d" % len(np.unique(annots[:, 0].astype('uint16'))), "%d" % len(annots)])

        for frame in np.unique(annots[:, 0].astype('uint16')):

            if frame == np.unique(annots[:, 0].astype('uint16'))[-1]:  # For Optical Flow
                continue

            frame_lines = annots[annots[:, 0] == str(frame)]
            if frame_lines.size > 0:

                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])

                f_objs = []
                for a in frame_lines:
                    cat_id = int(a[2]) - 1
                    if cat_id in classes_correspondence.values():
                        # cat_id = coco_correspondence[cat_id]
                        segm = {
                            "counts": a[-1].strip().encode(encoding='UTF-8'),
                            "size": [h, w]
                        }

                        box = coco.maskUtils.toBbox(segm)
                        box[2:] = box[2:] + box[:2]
                        box = box.tolist()

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
                            "category_id": cat_id,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "bbox": box,
                            "segmentation": poly

                        }
                        f_objs.append(annot)


                frame_data = {
                    "file_name": os.path.join(images_path, '{:06d}.{}'.format(int(a[0]), image_extension)),
                    "image_id": int(frame + n_seq * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": f_objs
                }
                mots_annots.append(frame_data)
    

    return mots_annots


def get_kitti_mots_instances_meta(dataset_name):
    thing_ids = [k["id"] for k in KITTI_MOTS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in KITTI_MOTS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 2, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in KITTI_MOTS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# def register_kitti_mots_instances(name, annots_path, imgs_path, train_percent=0.75, image_extension='png'):
def register_kitti_mots_instances(name, metadata, annots_path, imgs_path, train_percent=0.75, image_extension='png'):

    is_train = True if re.search("train", name) else False

    def get_kiti_mots_dicts_function(): return get_kiti_mots_dicts(name, imgs_path, annots_path, is_train=is_train,
                                                       train_percentage=train_percent, image_extension=image_extension)

    DatasetCatalog.register(name, get_kiti_mots_dicts_function)

    # MetadataCatalog.get(name).set(thing_classes=[k for k, v in classes_correspondence.items()])
    MetadataCatalog.get(name).set(
        image_root=imgs_path, evaluator_type="kitti", **metadata
    )

