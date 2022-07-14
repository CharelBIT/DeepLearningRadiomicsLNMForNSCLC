import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from mmcls.core.evaluation import precision_recall_f1, support, auc
from mmcls.models.losses import accuracy

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# def get_samples(root, folder_to_idx, extensions):
#     """Make dataset by walking all images under a root.
#
#     Args:
#         root (string): root directory of folders
#         folder_to_idx (dict): the map from class name to class idx
#         extensions (tuple): allowed extensions
#
#     Returns:
#         samples (list): a list of tuple where each element is (image, label)
#     """
#     samples = []
#     root = os.path.expanduser(root)
#     for folder_name in sorted(os.listdir(root)):
#         _dir = os.path.join(root, folder_name)
#         if not os.path.isdir(_dir):
#             continue
#
#         for _, _, fns in sorted(os.walk(_dir)):
#             for fn in sorted(fns):
#                 if has_file_allowed_extension(fn, extensions):
#                     path = os.path.join(folder_name, fn)
#                     item = (path, folder_to_idx[folder_name])
#                     samples.append(item)
#     return samples


@DATASETS.register_module()
class Hie_Dataset(BaseDataset):

    IMG_EXTENSIONS = ('.nii.gz')
    CLASSES = [
        '0', '1'
    ]

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 modes=[],
                 test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.modes = modes
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        if self.ann_file is None:
            print("[ERROR] need label info: {}".format(self.__class__.__name__))
        elif isinstance(self.ann_file, str):
            if self.ann_file.endswith('.txt'):
                with open(self.ann_file) as f:
                    samples = [x.strip().split(' ') for x in f.readlines()]
            else:
                raise NotImplementedError
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            filenames = []
            for mode in self.modes:
                filenames.append(os.path.join(self.data_prefix, filename, mode + '.nii.gz'))
            # info['patient_id'] = filename
            info['img_info'] = {'filename': filenames,
                                'patient_id': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

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
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
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
        eval_results['gt_labels'] = gt_labels.tolist()
        return eval_results
