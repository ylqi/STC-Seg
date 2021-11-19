import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

from .datasets.kitti import get_kitti_instances_meta, register_kitti_instances
from .datasets.kitti_mots import register_kitti_mots_instances, get_kitti_mots_instances_meta
from .datasets.kitti_mot import register_kitti_mot_instances, get_kitti_mot_instances_meta
from .datasets.ytvis import register_ytvis_instances, get_ytvis_instances_meta

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )





# ==== Predefined datasets and splits for YTVIS ==========


_PREDEFINED_SPLITS_YTVIS = {
    "ytvis_train": ("ytvis/train/JPEGImages", "ytvis/train.json"),
    "ytvis_val": ("ytvis/train/JPEGImages", "ytvis/valid.json"),
    "ytvis_sub_train": ("ytvis/train/JPEGImages", "ytvis/train_sub-train.json"),
    "ytvis_sub_val": ("ytvis/train/JPEGImages", "ytvis/train_sub-val.json"),
}



def register_all_ytvis(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            get_ytvis_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )





# ==== Predefined datasets and splits for KITTI ==========


_PREDEFINED_SPLITS_KITTI = {
    "kitti_object": {
        "kitti_object_train": ("kitti/training/image_2", "kitti/annotations/instances_train.json"),
        "kitti_object_val": ("kitti/training/image_2", "kitti/annotations/instances_val.json"),
    },
    "kitti_mots": {
        "kitti_mots_train": ("kitti_mots/training/image_02", "kitti_mots/instances_txt"),
        "kitti_mots_val": ("kitti_mots/training/image_02", "kitti_mots/instances_txt"),
        "kitti_mots_train_full": ("kitti_mots/training/image_02", "kitti_mots/instances_txt"),
    },
    "kitti_mot": {
        "kitti_mot_train": ("kitti_mot/training/image_02", "kitti_mot/training/label_02"),
        "kitti_mot_val": ("kitti_mot/training/image_02", "kitti_mot/training/label_02"),
    },
}


def register_all_kitti(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_KITTI.items():

        if dataset_name == "kitti_mots":
            for key, (image_root, txt_root) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_kitti_mots_instances(
                    key,
                    get_kitti_mots_instances_meta(dataset_name),
                    os.path.join(root, txt_root) if "://" not in txt_root else txt_root,
                    os.path.join(root, image_root),
                )

        elif dataset_name == "kitti_mot":
            for key, (image_root, txt_root) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_kitti_mot_instances(
                    key,
                    get_kitti_mot_instances_meta(dataset_name),
                    os.path.join(root, txt_root) if "://" not in txt_root else txt_root,
                    os.path.join(root, image_root),
                )

        else:
            for key, (image_root, json_file) in splits_per_dataset.items():
                # Assume pre-defined datasets live in `./datasets`.
                register_kitti_instances(
                    key,
                    get_kitti_instances_meta(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )



register_all_coco()

register_all_kitti()

register_all_ytvis()