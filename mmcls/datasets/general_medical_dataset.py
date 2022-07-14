import os

import numpy as np
import os.path as osp
from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from mmcls.core.evaluation import precision_recall_f1, support, auc
from mmcls.models.losses import accuracy

@DATASETS.register_module()
class GeneralMedicalDataset(BaseDataset):
    CLASSES = ("background", "1")
    def __init__(self,
                 pipeline,
                 img_dir,
                 split=None,
                 cross_validation_info=None,
                 img_suffixes=['image.nii.gz'],
                 mode='',
                 ann_dir=None,
                 seg_map_suffix="mask.nii.gz",
                 data_root=None,
                 test_mode=False,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 ignore_index=255):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffixes = img_suffixes
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.mode = mode
        self.CLASSES = self.get_classes(classes)
        self.ignore_index = ignore_index
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.data_infos = self.load_annotations(self.img_dir, self.mode, self.img_suffixes,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def load_annotations(self, img_dir, mode, img_suffixes, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_id, gt_label = line.strip().split(' ')
                    img_info = dict(img_info=dict())
                    img_info['img_info']['filename'] = []
                    img_info['patient_id'] = img_id
                    for img_suffix in img_suffixes:
                        img_name = osp.join(img_dir, img_id, mode, img_suffix)
                        img_info['img_info']['filename'].append(img_name)
                    if ann_dir is not None:
                        seg_map = osp.join(ann_dir, img_id, mode, seg_map_suffix)
                        img_info["ann_info"] = dict(seg_map=seg_map)
                    img_info['gt_label'] = np.array(gt_label, dtype=np.int64)
                    img_infos.append(img_info)
        else:
            raise NotImplementedError
        return img_infos

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        if 'auc' in metrics:
            auc_score = auc(results, gt_labels)
            eval_results['auc'] = auc_score

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results