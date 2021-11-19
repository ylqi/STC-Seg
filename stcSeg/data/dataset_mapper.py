import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

import re

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def save_tmp_image(image, tmp_dir="tmp", img_name=None):
    import os
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    if img_name is None:
        tmp_id = len(os.listdir(tmp_dir))
        img_name = "%d.png" % tmp_id

    import cv2
    cv2.imwrite("tmp/%s" % img_name, image)


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


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


def read_image_and_resize(file_name, shape, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray): an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)
        image = image.resize(shape)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = utils._apply_exif_orientation(image)

        return utils.convert_PIL_to_numpy(image, format)


def normalization(heatmap, target_min=-1, target_max=1):

    input_min = np.min(heatmap[np.nonzero(heatmap)])
    heatmap[np.nonzero(heatmap)] = heatmap[np.nonzero(heatmap)] - input_min

    input_max = np.max(heatmap)

    heatmap = heatmap / input_max * (target_max - target_min) + target_min

    return heatmap


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.stcseg_enabled = cfg.MODEL.STCSEG.ENABLED

        self.use_depth = cfg.MODEL.STCSEG.BOUNDARY.USE_DEPTH
        self.use_optical_flow = cfg.MODEL.STCSEG.BOUNDARY.USE_OPTICAL_FLOW

        if self.stcseg_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
            # print("%s image shape:" % dataset_dict["file_name"], image.shape)
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # save_tmp_image(image, img_name=dataset_dict["file_name"].split('/')[-1].split('.')[0] + '.png')

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))



        # ---------------------- Add Depth ---------------------------

        if self.use_depth: # For kitti object
            # print(dataset_dict["file_name"])
            try:
                if re.search("kitti_mot", dataset_dict["file_name"]):
                    # depth_image = utils.read_image(
                    #     dataset_dict["file_name"].replace("image_02", "image_depth").replace(".png", "_disp.jpeg"), format=self.image_format
                    # )
                    # depth_image = np.load(dataset_dict["file_name"].replace("training", "Depth/training").replace(".png", "_disp.npy"))
                    # depth_image = depth_image[0].transpose(1,2,0) * (1, 1, 1) * 10
                    depth_image = utils.read_image(
                        dataset_dict["file_name"].replace("image_02", "depth"), format=self.image_format
                    )
                    
                elif re.search("kitti", dataset_dict["file_name"]):
                    depth_image = utils.read_image(
                        dataset_dict["file_name"].replace("image_2", "depth"), format=self.image_format
                    )
                elif re.search("ytvis", dataset_dict["file_name"]):
                    depth_image = utils.read_image(
                        dataset_dict["file_name"].replace("JPEGImages", "Depth").replace(".jpg", ".png"), format=self.image_format
                    )
                    # print("%s depth shape:" % dataset_dict["file_name"], depth_image.shape)
                    # assert (depth_image.shape[1], depth_image.shape[0]) == (dataset_dict["width"], dataset_dict["height"]), dataset_dict["file_name"] + ": " + str(depth_image.shape)
                else:
                    print(dataset_dict["file_name"])
                    print("!!!!!!! Please use kitti or ytvis !!!!!!!")
            except Exception as e:
                print("Depth file for ", dataset_dict["file_name"])
                print(e)
                raise e
            try:
                utils.check_image_size(dataset_dict, depth_image)
            except SizeMismatchError as e:
                expected_wh = (dataset_dict["width"], dataset_dict["height"])
                depth_image_wh = (depth_image.shape[1], depth_image.shape[0])
                if (depth_image_wh[1], depth_image_wh[0]) == expected_wh:
                    print("transposing image {}".format(dataset_dict["file_name"]))
                    depth_image = depth_image.transpose(1, 0, 2)
                else:
                    raise e

            # aug_depth_input = T.StandardAugInput(depth_image, boxes=boxes, sem_seg=sem_seg_gt)
            # depth_transforms = aug_depth_input.apply_augmentations(self.augmentation)
            # depth_image = aug_depth_input.image
            depth_image = transforms.apply_image(depth_image)

            # save_tmp_image(depth_image, img_name=dataset_dict["file_name"].split('/')[-1].split('.')[0] + '_depth.png')

            dataset_dict["depth_image"] = torch.as_tensor(
                np.ascontiguousarray(depth_image.transpose(2, 0, 1))
            )

        # ---------------------- Add Depth ---------------------------


        # ---------------------- Add Flow ---------------------------

        if self.use_optical_flow: # For kitti object
            # print(dataset_dict["file_name"])
            try:
                if re.search("kitti_mot", dataset_dict["file_name"]):
                    flow_image_path = dataset_dict["file_name"].replace("image_02", "optical_flow")
                elif re.search("ytvis", dataset_dict["file_name"]):
                    flow_image_path = dataset_dict["file_name"].replace("JPEGImages", "OpticalFlow").replace(".jpg", ".png")                
                else:
                    print(dataset_dict["file_name"])
                    print("!!!!!!! Please use kitti mot or ytvis !!!!!!!")
                flow_image = read_image_and_resize(
                    flow_image_path, shape=(dataset_dict["width"], dataset_dict["height"]), 
                    format=self.image_format
                )
            except Exception as e:
                print(flow_image_path)
                print(e)
                raise e
            try:
                utils.check_image_size(dataset_dict, flow_image)
            except SizeMismatchError as e:
                expected_wh = (dataset_dict["width"], dataset_dict["height"])
                flow_image_wh = (flow_image.shape[1], flow_image.shape[0])
                if (flow_image_wh[1], flow_image_wh[0]) == expected_wh:
                    print("transposing image {}".format(dataset_dict["file_name"]))
                    flow_image = flow_image.transpose(1, 0, 2)
                else:
                    raise e

            # aug_flow_input = T.StandardAugInput(flow_image, boxes=boxes, sem_seg=sem_seg_gt)
            # flow_transforms = aug_flow_input.apply_augmentations(self.augmentation)
            # flow_image = aug_flow_input.image
            flow_image = transforms.apply_image(flow_image)

            # save_tmp_image(flow_image, img_name=dataset_dict["file_name"].split('/')[-1].split('.')[0] + '_flow.png')

            dataset_dict["flow_image"] = torch.as_tensor(
                np.ascontiguousarray(flow_image.transpose(2, 0, 1))
            )

        # ---------------------- Add Flow ---------------------------



        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
