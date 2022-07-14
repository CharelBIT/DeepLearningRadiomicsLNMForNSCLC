import copy
import warnings
import torch
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        self.test_cfg = test_cfg
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    # def inference(self, img, **kwargs):
    #     test_mode = self.test_cfg.get("mode", "whole")
    #     assert test_mode in ['slide', 'whole'], "[ERROR] Only Support one sample per batch"
    #
    # def slide_inference_3D(self, img, **kwargs):
    #     """Inference by sliding-window with overlap.
    #
    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.
    #     """
    #     x_stride, y_stride, z_stride = self.test_cfg.stride
    #     x_crop_length, y_crop_length, z_crop_length = self.test_cfg.crop_size
    #     batch_size, num_mode, x_img, y_img, z_img = img.size()
    #     pred_result = []
    #     x_grids = max(x_img - x_crop_length + x_stride - 1, 0) // x_stride + 1
    #     y_grids = max(y_img - y_crop_length + y_stride - 1, 0) // y_stride + 1
    #     z_grids = max(z_img - z_crop_length + z_stride - 1, 0) // z_stride + 1
    #     for x_idx in range(x_grids):
    #         for y_idx in range(y_grids):
    #             for z_idx in range(z_grids):
    #                 x1 = x_idx * x_stride
    #                 y1 = y_idx * y_stride
    #                 z1 = z_idx * z_stride
    #                 x2 = min(x1 + x_crop_length, x_img)
    #                 y2 = min(y1 + y_crop_length, y_img)
    #                 z2 = min(z1 + z_crop_length, z_img)
    #                 x1 = max(x2 - x_crop_length, 0)
    #                 y1 = max(y2 - y_crop_length, 0)
    #                 z1 = max(z2 - z_crop_length, 0)
    #                 crop_img = img[:, :, x1:x2, y1:y2, z1:z2]
    #                 x = self.extract_feat(crop_img)
    #                 x_dims = len(x.shape)
    #                 if x_dims == 1:
    #                     x.unsqueeze_(0)
    #                 pred_result.append(self.head.simple_test(x))
    #
    #     return preds
    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        # return self.inference(img, **kwargs)
        x = self.extract_feat(img)
        x_dims = len(x.shape)
        if x_dims == 1:
            x.unsqueeze_(0)
        return self.head.simple_test(x)
