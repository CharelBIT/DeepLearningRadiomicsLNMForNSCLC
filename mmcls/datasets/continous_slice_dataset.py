import os
import numpy as np
import os.path as osp
from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from mmcls.core.evaluation import precision_recall_f1, support, auc
from mmcls.models.losses import accuracy
from pycocotools import coco
import collections
@DATASETS.register_module()
class ContinousSliceDataset(BaseDataset):
    CLASSES = ("background", "1")
    def __init__(self,
                 pipeline,
                 img_dir,
                 split=None,
                 ann_file=None,
                 data_root=None,
                 test_mode=False,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 ignore_index=255):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES = self.get_classes(classes)
        self.ignore_index = ignore_index
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
        self.coco = coco.COCO(self.ann_file)
        # load annotations
        self.data_infos = self.load_annotations(self.img_dir,
                                               self.ann_file, self.split)

    def load_annotations(self, img_dir, ann_file, split):
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
        self.coco.createIndex()
        patient_id2image_id = collections.defaultdict(list)
        for img_id in self.coco.imgs:
            # patient_id2image_id[self.coco.imgs[img_id]['file_name'].split('.')[0]] = img_id
            patient_id2image_id[self.coco.imgs[img_id]['patient_id']].append(img_id)
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    patient_id, gt_label = line.strip().split(' ')
                    img_ids = patient_id2image_id[patient_id]
                    for img_id in img_ids:
                        img_info = dict(img_info=self.coco.imgs[img_id])
                        img_info['img_info']['filename'] = img_info['img_info']['file_name']
                        img_info['gt_label'] = np.array(gt_label, dtype=np.int64)
                        img_info['img_prefix'] = img_dir
                        img_info['ann_info'] = self.coco.imgToAnns[img_id]
                        img_infos.append(img_info)
        else:
            raise NotImplementedError
        return img_infos

    def get_patient_labels_and_preds(self, results, metric_options=None):
        assert len(results) == len(self.data_infos)
        patient_infos = {}
        for result, img_info in zip(results, self.data_infos):
            if img_info['img_info']['patient_id'] not in patient_infos:
                patient_infos[img_info['img_info']['patient_id']] = dict(pred=[])
            else:
                assert patient_infos[img_info['img_info']['patient_id']]['gt_label'] == img_info['gt_label']
            patient_infos[img_info['img_info']['patient_id']]['pred'].append(result)
            patient_infos[img_info['img_info']['patient_id']]['gt_label'] = img_info['gt_label']
        gt_per_patient = []
        pred_per_patient = []
        for patient_id in patient_infos:
            gt_per_patient.append(patient_infos[patient_id]['gt_label'])
            preds = patient_infos[patient_id]['pred']
            preds = np.vstack(preds)
            pred = np.mean(preds, axis=0)
            patient_infos[patient_id]['pred'] = pred
            pred_per_patient.append(pred)
        return pred_per_patient, gt_per_patient, patient_infos

    def evaluate(self,
               results,
               metric='accuracy',
               metric_options=None,
               logger=None):
        retVal = self.evaluate_per_slice(results, metric, metric_options, logger)
        retVal.update(self.evaluate_per_patient(results, metric, metric_options, logger))
        return retVal

    def evaluate_per_patient(self,
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
        results, gt_labels_per_patient, _ = self.get_patient_labels_and_preds(results, metric_options)
        results = np.vstack(results)
        gt_labels_per_patient = np.array(gt_labels_per_patient)
        num_imgs = len(results)
        assert len(gt_labels_per_patient) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels_per_patient, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels_per_patient, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'patient_accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'patient_accuracy': acc}
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
                results, gt_labels_per_patient, average_mode=average_mode)
            eval_results['patient_support'] = support_value

        if 'auc' in metrics:
            auc_score = auc(results, gt_labels_per_patient)
            eval_results['patient_auc'] = auc_score

        precision_recall_f1_keys = ['patient_precision', 'patient_recall', 'patient_f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels_per_patient, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels_per_patient, average_mode=average_mode)
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

    def evaluate_per_slice(self,
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
        # assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
        #                                    'be of the same length as gt_labels.'

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