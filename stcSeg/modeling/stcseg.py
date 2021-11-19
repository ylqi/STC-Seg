# -*- coding: utf-8 -*-
import logging
from skimage import color

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from stcSeg.utils.comm import aligned_bilinear

import os
import numpy as np
from PIL import Image
import re


__all__ = ["CondInst"]


logger = logging.getLogger(__name__)


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


def save_image(file_name, image, desc=""):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    original_image = Image.open(file_name)
    save_name = "%s_%s" % (file_name.split("/")[-2], file_name.split("/")[-1])

    image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # print(image.shape)
    image = Image.fromarray(image).resize(original_image.size)
    image.save(os.path.join(tmp_dir, save_name.replace(".", "_%s." % desc)))


def save_draft_image(file_name, cfg, images=None, color_similarity=None, depth_images=None, depth_images_color_similarity=None, flow_images=None, flow_images_color_similarity=None):
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    # print("images:", images.shape)
    # print("color_similarity", color_similarity.shape)
    # print("file_name", file_name)

    save_name = "%s_%s" % (file_name.split("/")[-2], file_name.split("/")[-1].replace('.jpg', '.png'))

    if not re.search("000204", file_name):
        return
    
    image = Image.open(file_name)
    image.save(os.path.join(tmp_dir, save_name))

    
    # if color_similarity is not None:
    #     for i, similarity_map in enumerate(color_similarity[0]):
    #         similarity_map = similarity_map.cpu().numpy()
    #         # similarity_map[similarity_map < 0.0001] = 0
    #         # similarity_map[similarity_map > 0.0001] = 0.1
    #         similarity_map = np.expand_dims(similarity_map, -1) * (255, 255, 255) * 10
    #         similarity_map[similarity_map > 255] = 255
    #         similarity_map = Image.fromarray(similarity_map.astype(np.uint8)).resize(image.size)
    #         similarity_map.save(os.path.join(tmp_dir, save_name.replace(".", "_col_sim_%d." % i)))



    if depth_images is not None:
        colored_depth = depth_images[0].cpu().numpy().transpose((1, 2, 0))
        colored_depth = (color.lab2rgb(colored_depth) * 255).astype(np.uint8)
        colored_depth = Image.fromarray(colored_depth).resize(image.size)
        colored_depth.save(os.path.join(tmp_dir, save_name.replace(".", "_dep.")))


    if depth_images_color_similarity is not None:
        for i, similarity_map in enumerate(depth_images_color_similarity[0]):
            similarity_map = similarity_map.cpu().numpy()
            # similarity_map[similarity_map < 0.3] = 0
            # similarity_map[similarity_map > 0.3] = 1
            similarity_map = np.expand_dims(similarity_map, -1) * (255, 255, 255) * 5
            similarity_map[similarity_map > 255] = 255
            similarity_map = Image.fromarray(similarity_map.astype(np.uint8)).resize(image.size)
            similarity_map.save(os.path.join(tmp_dir, save_name.replace(".", "_dep_sim_%d." % i)))




    if flow_images is not None:
        colored_flow = flow_images[0].cpu().numpy().transpose((1, 2, 0))
        colored_flow = (color.lab2rgb(colored_flow) * 255).astype(np.uint8)
        colored_flow = Image.fromarray(colored_flow).resize(image.size)
        colored_flow.save(os.path.join(tmp_dir, save_name.replace(".", "_flo.")))

    if flow_images_color_similarity is not None:
        for i, similarity_map in enumerate(flow_images_color_similarity[0]):
            similarity_map = similarity_map.cpu().numpy()
            # similarity_map[similarity_map < 0.4] = 0
            # similarity_map[similarity_map > 0.4] = 1
            similarity_map = np.expand_dims(similarity_map, -1) * (255, 255, 255) * 2
            similarity_map[similarity_map > 255] = 255
            similarity_map = Image.fromarray(similarity_map.astype(np.uint8)).resize(image.size)
            similarity_map.save(os.path.join(tmp_dir, save_name.replace(".", "_flo_sim_%d." % i)))



    if color_similarity is not None and depth_images_color_similarity is not None and flow_images_color_similarity is not None:
        for i, (color_similarity_map, depth_similarity_map) in enumerate(zip(color_similarity[0], depth_images_color_similarity[0])):
            color_similarity_map = color_similarity_map.cpu().numpy()
            color_similarity_map[color_similarity_map < 0.0001] = 0
            color_similarity_map[color_similarity_map > 0.0001] = 1

            depth_similarity_map = depth_similarity_map.cpu().numpy()
            depth_similarity_map[depth_similarity_map < 0.3] = 0
            depth_similarity_map[depth_similarity_map > 0.3] = 1

            for j, flow_similarity_map in enumerate(flow_images_color_similarity[0]):
                
                flow_similarity_map = flow_similarity_map.cpu().numpy()
                flow_similarity_map[flow_similarity_map < 0.4] = 0
                flow_similarity_map[flow_similarity_map > 0.4] = 1

                similarity_map = depth_similarity_map * flow_similarity_map

                similarity_map = np.expand_dims(similarity_map, -1) * (255, 255, 255)
                similarity_map[similarity_map > 255] = 255
                similarity_map = Image.fromarray(similarity_map.astype(np.uint8)).resize(image.size)
                similarity_map.save(os.path.join(tmp_dir, save_name.replace(".", "_X_%d_%d." % (i, j))))



@META_ARCH_REGISTRY.register()
class STCSeg(nn.Module):
    """
    Main class for STCSeg architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM

        # stcseg configs
        self.stcseg_enabled = cfg.MODEL.STCSEG.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.STCSEG.BOTTOM_PIXELS_REMOVED
        self.boundary_size = cfg.MODEL.STCSEG.BOUNDARY.SIZE
        self.boundary_dilation = cfg.MODEL.STCSEG.BOUNDARY.DILATION
        self.boundary_color_thresh = cfg.MODEL.STCSEG.BOUNDARY.COLOR_THRESH

        self.cfg = cfg

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        original_images = [x["image"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)
        # print("images_norm: ", len(images_norm))

        if "file_name" in batched_inputs[0]:
            file_names = [x["file_name"] for x in batched_inputs]


        # ----------------------------------- Add Depth ----------------------------------------
        if "depth_image" in batched_inputs[0]:
            original_depth_images = [x["depth_image"].to(self.device) for x in batched_inputs]

            # normalize images
            depth_images_norm = [self.normalizer(x) for x in original_depth_images]
            depth_images_norm = ImageList.from_tensors(depth_images_norm, self.backbone.size_divisibility)
        # ----------------------------------- Add Depth ----------------------------------------

        # ----------------------------------- Add Flow ----------------------------------------
        if "flow_image" in batched_inputs[0]:
            original_flow_images = [x["flow_image"].to(self.device) for x in batched_inputs]

            # normalize images
            flow_images_norm = [self.normalizer(x) for x in original_flow_images]
            flow_images_norm = ImageList.from_tensors(flow_images_norm, self.backbone.size_divisibility)
        # ----------------------------------- Add Flow ----------------------------------------


        features = self.backbone(images_norm.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.stcseg_enabled:
                original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]

                # mask out the bottom area where the COCO dataset probably has wrong annotations
                for i in range(len(original_image_masks)):
                    im_h = batched_inputs[i]["height"]
                    pixels_removed = int(
                        self.bottom_pixels_removed *
                        float(original_images[i].size(1)) / float(im_h)
                    )
                    if pixels_removed > 0:
                        original_image_masks[i][-pixels_removed:, :] = 0

                original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)
                original_image_masks = ImageList.from_tensors(
                    original_image_masks, self.backbone.size_divisibility, pad_value=0.0
                )

                # ----------------------------------- Add Depth ----------------------------------------
                if "depth_image" in batched_inputs[0]:
                    original_depth_images_tensor = ImageList.from_tensors(original_depth_images, self.backbone.size_divisibility).tensor
                else:
                    original_depth_images_tensor = None
                # ----------------------------------- Add Depth ----------------------------------------

                # ----------------------------------- Add Flow ----------------------------------------
                if "flow_image" in batched_inputs[0]:
                    original_flow_images_tensor = ImageList.from_tensors(original_flow_images, self.backbone.size_divisibility).tensor
                else:
                    original_flow_images_tensor = None
                # ----------------------------------- Add Flow ----------------------------------------

                self.add_bitmasks_from_boxes(
                    gt_instances, original_images.tensor, original_image_masks.tensor,
                    original_images.tensor.size(-2), original_images.tensor.size(-1),
                    original_depth_images_tensor, original_flow_images_tensor, file_names
                )
            else:
                self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller
        )

        if self.training:
            mask_losses = self._forward_mask_heads_train(proposals, mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w, depth_images=None, flow_images=None, file_names=None):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        # print("image shape:", images.size())
        # print("depth shape:", depth_images.size())

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]]
        image_masks = image_masks[:, start::stride, start::stride]

        # ------------ Add Depth ----------------
        if depth_images is not None:
            downsampled_depth_images = F.avg_pool2d(
                depth_images.float(), kernel_size=stride,
                stride=stride, padding=0
            )[:, [2, 1, 0]]
        # ------------ Add Depth ----------------

        # ------------ Add Flow ----------------
        if flow_images is not None:
            downsampled_flow_images = F.avg_pool2d(
                flow_images.float(), kernel_size=stride,
                stride=stride, padding=0
            )[:, [2, 1, 0]]
        # ------------ Add Flow ----------------

        for im_i, per_im_gt_inst in enumerate(instances):

            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
            # save_image(file_names[im_i], downsampled_images[im_i], desc="downsampled")
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.boundary_size, self.boundary_dilation
            )
            # print("downsampled image shape:", images_lab.size())

            # ------------ Add Depth ----------------
            if depth_images is not None:
                depth_images_lab = color.rgb2lab(downsampled_depth_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
                depth_images_lab = torch.as_tensor(depth_images_lab, device=downsampled_depth_images.device, dtype=torch.float32)
                depth_images_lab = depth_images_lab.permute(2, 0, 1)[None]
                # depth_images_lab = downsampled_depth_images[im_i][None]

                depth_images_color_similarity = get_images_color_similarity(
                    depth_images_lab, image_masks[im_i],
                    self.boundary_size, self.boundary_dilation
                )
                # print("downsampled depth shape:", depth_images_lab.size())
            # ------------ Add Depth ----------------

            # ------------ Add Flow ----------------
            if flow_images is not None:
                flow_images_lab = color.rgb2lab(downsampled_flow_images[im_i].byte().permute(1, 2, 0).cpu().numpy())
                flow_images_lab = torch.as_tensor(flow_images_lab, device=downsampled_flow_images.device, dtype=torch.float32)
                flow_images_lab = flow_images_lab.permute(2, 0, 1)[None]
                flow_images_color_similarity = get_images_color_similarity(
                    flow_images_lab, image_masks[im_i],
                    self.boundary_size, self.boundary_dilation
                )
            # ------------ Add Flow ----------------

            # save_draft_image(file_names[im_i], self.cfg, images_lab, images_color_similarity, 
            #                  depth_images_lab, depth_images_color_similarity, 
            #                  flow_images_lab, flow_images_color_similarity)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w)).to(self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)
            
            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst))
            ], dim=0)

            # ------------ Add Depth ----------------
            if depth_images is not None:
                per_im_gt_inst.depth_image_color_similarity = torch.cat([
                    depth_images_color_similarity for _ in range(len(per_im_gt_inst))
                ], dim=0)
            # ------------ Add Depth ----------------

            # ------------ Add Flow ----------------
            if flow_images is not None:
                per_im_gt_inst.flow_image_color_similarity = torch.cat([
                    flow_images_color_similarity for _ in range(len(per_im_gt_inst))
                ], dim=0)
            # ------------ Add Flow ----------------


    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
