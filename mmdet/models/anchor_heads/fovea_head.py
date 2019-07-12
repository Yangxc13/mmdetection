import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import multi_apply, multiclass_nms, distance2bbox
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8
debug = 0


@HEADS.register_module
class FoveaHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 126, 256),
                 scale_ranges=((8,32), (16,64), (32,128), (64,256), (128,512)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None):
        super(FoveaHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fovea_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fovea_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fovea_cls, std=0.01, bias=bias_cls)
        normal_init(self.fovea_reg, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fovea_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.fovea_reg(reg_feat) # 相比于fcos，去掉了.exp()

        return cls_score, bbox_pred


    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        points = []
        for featmap_size in featmap_sizes:
            x_range = torch.arange(featmap_size[1], dtype=dtype, device=device) + 0.5
            y_range = torch.arange(featmap_size[0], dtype=dtype, device=device) + 0.5
            y, x = torch.meshgrid(y_range, x_range)
            if flatten:
                points.append((y.flatten(), x.flatten()))
            else:
                points.append((y,x))

        return points

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bbox_list, # (x1, y1, x2, y2)
             gt_label_list, # 1~num_classes
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # [[152, 100], [76, 50], [38, 25], [19, 13], [10, 7]]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                            bbox_preds[0].device)

        label_list, bbox_target_list = multi_apply(
            self.fovea_target_single,
            gt_bbox_list,
            gt_label_list,
            featmap_size_list=featmap_sizes,
            point_list=points)

        # 1. 该步骤已经自动分配了label>-1,label==0和label==-1的区域，因此不在需要assigner
        # 2. 因为使用FocalLoss，所以不需要使用sampler
        flatten_labels = [
            torch.cat([labels_level_img.flatten()
                for labels_level_img in labels_level])
            for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([bbox_targets_level_img.reshape(-1, 4)
                for bbox_targets_level_img in bbox_targets_level])
            for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) # [n,c,h,w]->[n*h*w,c]
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        pos_inds = (flatten_labels > 0).nonzero().view(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0 （详见笔记7-1.1）

        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                # weight=pos_centerness_targets,  这里是一个可能对最终结果很重要的不同。不像fcos_head，fovea_head没有权重的区别
                avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(0, dtype=flatten_bbox_preds.dtype, device=flatten_bbox_preds.device)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)

    def fovea_target_single(self, gt_bboxes_raw, gt_labels_raw,
        sigma1=0.3, sigma2=0.4, featmap_size_list=None, point_list=None):

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
            gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        if debug:
            if torch.any(gt_areas < self.scale_ranges[0][0]):
                print('Warning: too small targets', gt_bboxes_raw[gt_areas < self.scale_ranges[0][0]])
            if torch.any(gt_areas > self.scale_ranges[-1][1]):
                print('Warning: too large targets', gt_bboxes_raw[gt_areas < self.scale_ranges[-1][1]])

        label_list = []
        bbox_target_list = []
        for base_len, (lower_bound, upper_bound), stride, featmap_size, (y,x) \
            in zip(self.base_edge_list, self.scale_ranges, self.strides, featmap_size_list, point_list):
            labels = gt_labels_raw.new_zeros(featmap_size)
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1], 4) + 1

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue

            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order] # 面积大的排在前面

            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]

            half_w = 0.5 * (gt_bboxes[:,2] - gt_bboxes[:,0])
            half_h = 0.5 * (gt_bboxes[:,3] - gt_bboxes[:,1])

            pos_left = torch.ceil(gt_bboxes[:,0] + (1 - sigma1) * half_w - 0.5).long().clamp(0,featmap_size[1]-1)
            pos_right = torch.floor(gt_bboxes[:,0] + (1 + sigma1) * half_w - 0.5).long().clamp(0,featmap_size[1]-1)
            pos_top = torch.ceil(gt_bboxes[:,1] + (1 - sigma1) * half_h - 0.5).long().clamp(0,featmap_size[0]-1)
            pos_down = torch.floor(gt_bboxes[:,1] + (1 + sigma1) * half_h - 0.5).long().clamp(0,featmap_size[0]-1)

            neg_left = torch.ceil(gt_bboxes[:,0] + (1 - sigma2) * half_w - 0.5).long().clamp(0,featmap_size[1]-1)
            neg_right = torch.floor(gt_bboxes[:,0] + (1 + sigma2) * half_w - 0.5).long().clamp(0,featmap_size[1]-1)
            neg_top = torch.ceil(gt_bboxes[:,1] + (1 - sigma2) * half_h - 0.5).long().clamp(0,featmap_size[0]-1)
            neg_down = torch.floor(gt_bboxes[:,1] + (1 + sigma2) * half_h - 0.5).long().clamp(0,featmap_size[0]-1)

            for px1, py1, px2, py2, nx1, ny1, nx2, ny2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                zip(pos_left, pos_top, pos_right, pos_down, neg_left, neg_top, neg_right, neg_down, gt_labels, gt_bboxes_raw[hit_indices,:]):

                # negative: top
                labels[ny1:py1, px1:px2+1][labels[ny1:py1, px1:px2+1] == 0] = -1
                # bottom
                labels[py2+1:ny2+1, px1:px2+1][labels[py2+1:ny2+1, px1:px2+1] == 0] = -1
                # left
                labels[ny1:ny2+1, nx1:px1][labels[ny1:ny2+1, nx1:px1] == 0] = -1
                # right
                labels[ny1:ny2+1, px2+1:nx2+1][labels[ny1:ny2+1, px2+1:nx2+1] == 0] = -1
                # positive:
                labels[py1:py2+1, px1:px2+1] = label

                flag = False
                if torch.any(stride * x[py1:py2+1, px1:px2+1] - gt_x1 <= 1e-6):
                    print('cx-x1 error', (gt_x1, gt_y1, gt_x2, gt_y2), stride * x[py1:py2+1, px1:px2+1], gt_x1)
                    flag = True
                if torch.any(stride * y[py1:py2+1, px1:px2+1] - gt_y1 <= 1e-6):
                    print('cy-y1 error', (gt_x1, gt_y1, gt_x2, gt_y2), stride * y[py1:py2+1, px1:px2+1], gt_y1)
                    flag = True
                if torch.any(gt_x2 - stride * x[py1:py2+1, px1:px2+1] <= 1e-6):
                    print('x2-cx error', (gt_x1, gt_y1, gt_x2, gt_y2), gt_x2, stride * x[py1:py2+1, px1:px2+1])
                    flag = True
                if torch.any(gt_y2 - stride * y[py1:py2+1, px1:px2+1] <= 1e-6):
                    print('y2-cy error', (gt_x1, gt_y1, gt_x2, gt_y2), gt_y2, stride * y[py1:py2+1, px1:px2+1])
                    flag = True

                bbox_targets[py1:py2+1, px1:px2+1, 0] = (stride * x[py1:py2+1, px1:px2+1] - gt_x1) / base_len
                bbox_targets[py1:py2+1, px1:px2+1, 1] = (stride * y[py1:py2+1, px1:px2+1] - gt_y1) / base_len
                bbox_targets[py1:py2+1, px1:px2+1, 2] = (gt_x2 - stride * x[py1:py2+1, px1:px2+1]) / base_len
                bbox_targets[py1:py2+1, px1:px2+1, 3] = (gt_y2 - stride * y[py1:py2+1, px1:px2+1]) / base_len

                if flag: bbox_targets[bbox_targets <= 1e-6] = 1

            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))

        return label_list, bbox_target_list

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg,
                   rescale=None): # only for test
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                            bbox_preds[0].device, flatten=True)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list, featmap_sizes, points,
                                                img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          featmap_sizes,
                          point_list,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False, debug=False):
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y,x) in zip(
                cls_scores, bbox_preds, featmap_sizes, self.strides, self.base_edge_list, point_list):
            if debug:
                if torch.max(cls_score.flatten()) < 1: continue
                scores = cls_score.reshape(-1, self.cls_out_channels).float()
                bbox_pred = bbox_pred.reshape(-1, 4).exp()
                topk_inds = scores.max(dim=-1)[0] > 0.5
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y.flatten()[topk_inds]
                x = x.flatten()[topk_inds]
            else:
                assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                scores = cls_score.permute(1, 2, 0).reshape(
                    -1, self.cls_out_channels).sigmoid()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
                nms_pre = cfg.get('nms_pre', -1)
                if nms_pre > 0 and scores.shape[0] > nms_pre:
                    max_scores, _ = scores.max(dim=1)
                    _, topk_inds = max_scores.topk(nms_pre)
                    bbox_pred = bbox_pred[topk_inds, :]
                    scores = scores[topk_inds, :]
                    y = y[topk_inds]
                    x = x[topk_inds] # 非debug模式下，已经实现flatten好了
            x1 = (stride * x - base_len * bbox_pred[:,0]).clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:,1]).clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:,2]).clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:,3]).clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
        det_bboxes = torch.cat(det_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        det_scores = torch.cat([padding, det_scores], dim=1)
        if debug:
            det_bboxes, det_labels = multiclass_nms(
                det_bboxes,
                det_scores,
                cfg['score_thr'],
                cfg['nms'],
                cfg['max_per_img'])
        else:
            det_bboxes, det_labels = multiclass_nms(
                det_bboxes,
                det_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
        return det_bboxes, det_labels

