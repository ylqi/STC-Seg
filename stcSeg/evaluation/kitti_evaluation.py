import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

from tqdm import tqdm
import sys
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco

from detectron2.data.detection_utils import read_image
from stcSeg.utils.tracking import track_objects, save_instance


sys.path.append('../mots_tools')
import pycocotools.mask as rletools


score_threshold = [0.47,  0.58]

class Predictions(object):
    def __init__(self):
        self.pred_masks = []
        self.pred_classes = []
        self.pred_boxes = torch.Tensor([])
        self.scores = []

        self.keys = []

    def add(self, bbox, mask, category_id, score):
        self.pred_masks.append(mask)
        self.pred_classes.append(category_id)
        if isinstance(bbox, list):
            bbox = torch.Tensor(bbox)
        if len(self.pred_boxes) == 0:
            self.pred_boxes = bbox.unsqueeze(0)
        else:
            self.pred_boxes = torch.cat((self.pred_boxes, bbox.unsqueeze(0)), 0)
        self.scores.append(score)

        self.keys = ["pred_masks", "pred_classes", "pred_boxes", "scores"]


    def has(self, key):
        return key in self.keys


class KittiEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            new_coco_results = []
            for result in coco_results:
                # print("image_id:", result["image_id"])
                category_id = result["category_id"]
                # assert (
                #     category_id in reverse_id_mapping
                # ), "A prediction has category_id={}, which is not available in the dataset.".format(
                #     category_id
                # )
                if category_id not in reverse_id_mapping:
                    # print("A prediction has category_id={}, which is not available in the dataset.".format(category_id))
                    continue
                result["category_id"] = reverse_id_mapping[category_id]
                new_coco_results.append(result)
            coco_results = new_coco_results

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res


        self._logger.info("Start evaluation on KITTI MOTS")


        instance_txt_out_dir="instances_txt/eval"
        os.system("rm -rf %s" % instance_txt_out_dir)
        os.makedirs(instance_txt_out_dir, exist_ok=True)


        predictions_dict = {}

        self._logger.info("Converting coco format into KITTI format:")

        for result in tqdm(coco_results):
            image_id = result["image_id"]

            seq_name = "%04d" % int(image_id / 1000000)
            image_name = "%06d" % int(image_id % 1000000)

            if seq_name not in predictions_dict.keys():
                predictions_dict[seq_name] = {}

            if image_name not in predictions_dict[seq_name].keys():
                predictions_dict[seq_name][image_name] = Predictions()

            category_id = result["category_id"] - 1  # 1: Car  2: Pedestrain
            score = result["score"]
            segmentation = result["segmentation"]
            mask = rletools.decode(segmentation)
            bbox = result["bbox"]

            if category_id >= len(self._metadata.get("thing_classes")):
                continue

            if score < score_threshold[category_id]:
                continue

            predictions_dict[seq_name][image_name].add(bbox, mask, category_id, score)

        self._logger.info("Start multi-object tracking on videos:")

        for seq_name in predictions_dict.keys():
            
            tracking_dict = {}

            for image_name in tqdm(predictions_dict[seq_name].keys(), desc=seq_name):

                path = 'datasets/kitti_mots/training/image_02/%s/%s.png' % (seq_name, image_name)

                img = read_image(path, format="BGR")

                depth_file_path = 'datasets/kitti_mots/training/depth/%s/%s.npy' % (seq_name, image_name)
                depth_data = np.load(depth_file_path)
                depth_data = depth_data.transpose(1,2,0)

                
                flow_data = None
                # flow_file_path = 'datasets/kitti_mots/training/optical_flow/%s/%s.flo' % (seq_name, image_name)
                # f = open(flow_file_path, 'rb')
                # x = np.fromfile(f, np.int32, count=1) # not sure what this gives
                # w = int(np.fromfile(f, np.int32, count=1)) # width
                # h = int(np.fromfile(f, np.int32, count=1)) # height
                # flow_data = np.fromfile(f, np.float32) # vector 
                # flow_data = np.reshape(flow_data, newshape=(h, w, 2)); # convert to x,y - flow
                # flow_data = scipy.ndimage.zoom(flow_data, (img.shape[0] / h, img.shape[1] / w, 1))

                predictions = predictions_dict[seq_name][image_name]

                tracking_dict, track_id_list, assigned_colors, info_list = track_objects(predictions=predictions, tracking_dict=tracking_dict, 
                                                      depth_data=depth_data, flow_data=flow_data, box_mode="XYXY_ABS")

                save_instance(tracking_dict=tracking_dict, path=path, class_names=self._metadata.get("thing_classes"), 
                                                    instance_txt_out_dir=instance_txt_out_dir, dataset="KITTI_MOTS")

        now_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self._logger.info("Evaluation provided by mots_tools (CLEAR MOT metrics):")
        motsa_save_path = os.path.join(self._output_dir, "scores_%s_MOTSA.txt" % now_time)
        os.system("cd ../mots_tools && ./run_eval.sh ../STC-Seg/%s | tee ../STC-Seg/%s" % (instance_txt_out_dir, motsa_save_path))

        self._logger.info("Evaluation provided by TrackEval (HOTA tracking metrics):")
        hota_save_path = os.path.join(self._output_dir, "scores_%s_HOTA.txt" % now_time)
        os.system("cd ../TrackEval && ./run_eval.sh ../STC-Seg/%s | tee ../STC-Seg/%s" % (instance_txt_out_dir, hota_save_path))

        self._logger.info("Saving results to %s and %s" % (motsa_save_path, hota_save_path))

            
            


