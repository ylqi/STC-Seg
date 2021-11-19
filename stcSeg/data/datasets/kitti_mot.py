# The label files contain the following information, which can be read and
# written using the matlab tools (readLabels.m) provided within this devkit. 
# All values (numerical or strings) are separated via spaces, each row 
# corresponds to one object. The 17 columns represent:

# #Values    Name      Description
# ----------------------------------------------------------------------------
#    1    frame        Frame within the sequence where the object appearers
#    1    track id     Unique tracking id of this object within this sequence
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Integer (0,1,2) indicating the level of truncation.
#                      Note that this is in contrast to the object detection
#                      benchmark where truncation is a float in [0,1].
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.


import os
import cv2
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco
import numpy as np


cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person',
                'Tram']

# cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person',
#                 'Tram', 'Misc']



KITTI_MOT_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": cats[0]},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": cats[1]},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": cats[2]},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": cats[3]},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": cats[4]},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": cats[5]},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": cats[6]},
]
# KITTI_MOT_CATEGORIES = [
#     {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": cats[0]},
#     {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": cats[1]},
#     {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": cats[2]},
#     {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": cats[3]},
#     {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": cats[4]},
#     {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": cats[5]},
#     {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": cats[6]},
#     {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": cats[7]},
# ]


def get_kitti_mot_dicts(images_folder, annots_folder, is_train, train_percentage=0.75, image_extension="png"):
    assert os.path.exists(images_folder), images_folder
    assert os.path.exists(annots_folder)

    # annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))

    # n_train_seqences = int(len(annot_files) * train_percentage)
    # train_sequences = annot_files[:n_train_seqences]
    # test_sequences = annot_files[n_train_seqences:]

    # sequences = train_sequences if is_train else test_sequences

    split = "train" if is_train else "val"

    annot_files = []
    seqmap_file = open("datasets/kitti_mot/splits/%s.seqmap" % (split), 'r')
    for line in seqmap_file:
        fields = line.split(" ")
        annot_files.append(os.path.join(annots_folder, "%04d.txt" % int(fields[0])))

    # --------------------------- Use all to train -------------------------
    if is_train and re.search("full", name):
        annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))
    # --------------------------- Use all to train -------------------------

    sequences = annot_files

    kitti_mot_annotations = []
    for seq_file in sequences:
        seq_images_path = os.path.join(images_folder, seq_file.split("/")[-1].split(".")[0])
        kitti_mot_annotations += mot_annots_to_coco(seq_images_path, seq_file, image_extension)

    return kitti_mot_annotations


def mot_annots_to_coco(images_path, txt_file, image_extension):
    assert os.path.exists(txt_file)
    n_seq = int(txt_file.split("/")[-1].split(".")[0])

    mot_annots = []
    with open(txt_file, 'r') as f:
        annots = f.readlines()
        annots = [l.split() for l in annots]

        annots = np.array(annots)

        for frame in np.unique(annots[:, 0].astype('uint8')):

            if frame == np.unique(annots[:, 0].astype('uint8'))[-1]:  # For Optical Flow
                continue

            frame_lines = annots[annots[:, 0] == str(frame)]
            if frame_lines.size > 0:

                # h, w = int(frame_lines[0][3]), int(frame_lines[0][4])
                img = cv2.imread(os.path.join(images_path, '{:06d}.{}'.format(int(frame_lines[0][0]), image_extension)))
                h, w, _ = img.shape

                f_objs = []
                for a in frame_lines:
                    cat_name = a[2]
                    # if cat_name == "DontCare":
                    #     continue
                    if cat_name == "DontCare" or cat_name == "Misc":
                        continue
                    cat_id = cats.index(cat_name)

                    box = [int(round(float(a[6]))), int(round(float(a[7]))), int(round(float(a[8]))), int(round(float(a[9]))),]
                    # print(box)

                    annot = {
                        "category_id": cat_id,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "bbox": box,
                        "segmentation": []

                    }
                    f_objs.append(annot)


                frame_data = {
                    "file_name": os.path.join(images_path, '{:06d}.{}'.format(int(a[0]), image_extension)),
                    "image_id": int(frame + n_seq * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": f_objs
                }
                mot_annots.append(frame_data)

    return mot_annots


def get_kitti_mot_instances_meta(dataset_name):
    thing_ids = [k["id"] for k in KITTI_MOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in KITTI_MOT_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 7, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in KITTI_MOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# def register_kitti_mot_instances(name, annots_path, imgs_path, train_percent=0.75, image_extension='png'):
def register_kitti_mot_instances(name, metadata, annots_path, imgs_path, train_percent=0.75, image_extension='png'):

    is_train = True if name == "kitti_mot_train" else False

    def get_kitti_mot_dicts_function(): return get_kitti_mot_dicts(imgs_path, annots_path, is_train=is_train,
                                                       train_percentage=train_percent, image_extension=image_extension)

    DatasetCatalog.register(name, get_kitti_mot_dicts_function)

    # MetadataCatalog.get(name).set(thing_classes=[k for k, v in classes_correspondence.items()])
    MetadataCatalog.get(name).set(
        image_root=imgs_path, evaluator_type="kitti", **metadata
    )

