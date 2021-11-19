import torch
from torch.nn import functional as F
from torch import nn

import numpy as np

from stcSeg.utils.comm import compute_locations, aligned_bilinear
from stcSeg.utils.comm import reduce_sum, reduce_mean, compute_ious
from stcSeg.layers import ml_nms, IOULoss


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)

def find_first_value(scores, threshold=0.3):
    for i, v in enumerate(scores):
        if v > threshold:
            return i
    return None


def compute_box_term(mask_scores, gt_bitmasks, instances=None, loss_type="r-Dice"):

    if loss_type == "Dice":
        mask_losses_y = dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean()

    elif loss_type == "r-Dice":
        mask_losses_y = r_dice_coefficient(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = r_dice_coefficient(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        return (mask_losses_x + mask_losses_y).mean()

    elif loss_type == "GIoU":
        print("mask_scores:", mask_scores.shape)
        print("gt_bitmasks:", gt_bitmasks.shape)
        print("instances:", instances.reg_pred.size())

        num_classes = instances.logits_pred.size(1)

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)

        num_pos_local = torch.ones_like(pos_inds).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

        ctrness_targets = compute_ctrness_targets(instances.reg_targets)

        scores_y = mask_scores.max(dim=2, keepdim=True)[0].reshape(-1)
        scores_x = mask_scores.max(dim=2, keepdim=True)[0].reshape(-1)
        # print(scores_x)
        left = find_first_value(scores_x)
        top = find_first_value(scores_y)
        right = find_first_value(torch.flip(scores_x.reshape(scores_x.shape[0],1),[0,1]).reshape(-1))
        bottom = find_first_value(torch.flip(scores_y.reshape(scores_y.shape[0],1),[0,1]).reshape(-1))
        mask_bbox = torch.Tensor([left, top, right, bottom])

        ious, gious = compute_ious(mask_bbox, instances.reg_targets)
        loc_loss_func = IOULoss('giou')
        reg_loss = loc_loss_func(ious, gious, ctrness_targets) / num_pos_avg
        return reg_loss

    else:
        raise "loss type error."


def compute_boundary_term(mask_logits, boundary_size, boundary_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=boundary_size,
        dilation=boundary_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=boundary_size,
        dilation=boundary_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def r_dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    # print("x:", x)
    target = target.reshape(n_inst, -1)
    # print("target:", target)
    intersection = (x * target).sum(dim=1)
    # print("intersection: ", intersection)
    # outsection = (x ** 2.0).sum(dim=1) - intersection
    # outsection[outsection < 0] = 0
    # print("outsection:", outsection)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    outsection = x - target
    outsection[outsection < 0] = 0
    loss = 1. - (2 * intersection / union) + (outsection ** 2.0).sum(dim=1) / ((target ** 2.0).sum(dim=1) + eps)
    # print(loss)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # stcseg configs
        self.stcseg_enabled = cfg.MODEL.STCSEG.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.STCSEG.BOTTOM_PIXELS_REMOVED
        self.boundary_size = cfg.MODEL.STCSEG.BOUNDARY.SIZE
        self.boundary_dilation = cfg.MODEL.STCSEG.BOUNDARY.DILATION
        self.boundary_color_thresh = cfg.MODEL.STCSEG.BOUNDARY.COLOR_THRESH

        self._warmup_iters = cfg.MODEL.STCSEG.BOUNDARY.WARMUP_ITERS

        self.use_depth = cfg.MODEL.STCSEG.BOUNDARY.USE_DEPTH
        self.use_optical_flow = cfg.MODEL.STCSEG.BOUNDARY.USE_OPTICAL_FLOW
        self.boundary_depth_color_thresh = cfg.MODEL.STCSEG.BOUNDARY.DEPTH_COLOR_THRESH
        self.boundary_flow_color_thresh = cfg.MODEL.STCSEG.BOUNDARY.FLOW_COLOR_THRESH
        self.boundary_final_thresh = cfg.MODEL.STCSEG.BOUNDARY.FINAL_THRESH
        self.boundary_similarity_method = cfg.MODEL.STCSEG.BOUNDARY.SIMILARITY_METHOD
        
        self.use_bce_loss = cfg.MODEL.STCSEG.BCE_LOSS.ENABLED

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)
        # print("n_inst:", n_inst)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            # print("pred_instances: ", len(pred_instances))
            # print("img_num: ", len(gt_instances))

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.stcseg_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_box"] = dummy_loss
                    losses["loss_boundary"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()

                if self.stcseg_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    # ----------------------------------------- Add Depth ---------------------------------------------
                    if self.use_depth:
                        depth_image_color_similarity = torch.cat([x.depth_image_color_similarity for x in gt_instances])
                        depth_image_color_similarity = depth_image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)
                    # ----------------------------------------- Add Depth ---------------------------------------------

                    # ----------------------------------------- Add Flow ---------------------------------------------
                    if self.use_optical_flow:
                        flow_image_color_similarity = torch.cat([x.flow_image_color_similarity for x in gt_instances])
                        flow_image_color_similarity = flow_image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)
                    # ----------------------------------------- Add Flow ---------------------------------------------

                    loss_box_term = compute_box_term(mask_scores, gt_bitmasks, pred_instances[gt_inds])

                    boundary_losses = compute_boundary_term(
                        mask_logits, self.boundary_size,
                        self.boundary_dilation
                    )

                    weights = gt_bitmasks.float()
                    # print("weights: ", weights.shape)

                    similarity = image_color_similarity

                    if self.boundary_similarity_method == "X":
                        weights = weights * (image_color_similarity >= self.boundary_color_thresh).float()
                        if self.use_depth:
                            weights = weights * (depth_image_color_similarity >= self.boundary_depth_color_thresh).float()
                        if self.use_optical_flow:
                            weights = weights * (flow_image_color_similarity >= self.boundary_flow_color_thresh).float() 
                    elif self.boundary_similarity_method == "XR":
                        assert self.use_depth and self.use_optical_flow
                        # print(weights.shape)
                        weights = weights * (((image_color_similarity >= self.boundary_color_thresh).float() 
                                             * (depth_image_color_similarity >= self.boundary_depth_color_thresh).float())
                                             + ((image_color_similarity >= self.boundary_color_thresh).float()
                                             * (flow_image_color_similarity >= self.boundary_flow_color_thresh).float())
                                             )
                        # print(weights.shape)
                    else:
                        if self.use_depth:
                            if self.boundary_similarity_method == "PLUS":
                                similarity = similarity * depth_image_color_similarity
                            elif self.boundary_similarity_method == "ADD":
                                similarity = similarity + depth_image_color_similarity
                        if self.use_optical_flow:
                            if self.boundary_similarity_method == "PLUS":
                                similarity = similarity * flow_image_color_similarity
                            elif self.boundary_similarity_method == "ADD":
                                similarity = similarity + flow_image_color_similarity

                        weights = weights * (similarity >= self.boundary_final_thresh).float()
                    
                    if self.use_bce_loss:
                        # print(np.max(mask_scores.detach().cpu().numpy()))
                        nagative_boundary_losses = compute_boundary_term(
                            -mask_logits, self.boundary_size,
                            self.boundary_dilation
                        )
                        loss_boundary = (boundary_losses * weights).sum() / weights.sum().clamp(min=1.0) + (nagative_boundary_losses * (1 - weights) * gt_bitmasks.float()).sum() / (1 - weights).sum().clamp(min=1.0)
                    else:
                        loss_boundary = (boundary_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_boundary = loss_boundary * warmup_factor

                    losses.update({
                            "loss_box": loss_box_term,
                            "loss_boundary": loss_boundary,
                        })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances
