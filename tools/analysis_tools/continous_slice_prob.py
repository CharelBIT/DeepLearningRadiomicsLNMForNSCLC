import argparse
import os
import warnings
import mmcv
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default=None, help='output result file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_model_prob(model, data_loader):
    results = []
    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            results.extend(result)
    return results

def plot_figs(pred_prob, gt_label, dataset='val'):
    if isinstance(pred_prob, list):
        pred_prob = np.asarray(pred_prob)
    if isinstance(gt_label, list):
        gt_label = np.asarray(gt_label)
    fpr, tpr, th = metrics.roc_curve(gt_label, pred_prob[:, 1])
    auc = metrics.auc(fpr, tpr)
    plot_roc(fpr, tpr, th, auc, dataset)
    plot_dca(gt_label, pred_prob[:, 1], dataset)


def plot_roc(fpr, tpr, th, auc, dataset):
    uindex = np.argmax(tpr - fpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.plot(fpr[uindex], tpr[uindex], 'r', markersize=8)
    plt.text(fpr[uindex], tpr[uindex], '%.3f(%.3f,%.3f)' % (th[uindex], fpr[uindex], tpr[uindex]), ha='center',
             va='bottom', fontsize=10)
    plt.title('ROC curve (' + dataset + ')')
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(True)
    plt.show()

def plot_dca(y_true, y_prob, dataset):
    thresholds = np.array([i / 100.0 for i in range(100)])
    y_prob_0 = np.zeros(y_prob.shape)
    y_prob_1 = np.ones(y_prob.shape)
    net_benefits = np.zeros(100)
    net_benefits_0 = np.zeros(100)  # None
    net_benefits_1 = np.zeros(100)  # All
    for i in range(100):
        th = thresholds[i]
        net_benefits[i] = netBenefit(y_true, y_prob, th)
        net_benefits_0[i] = netBenefit(y_true, y_prob_0, th)
        net_benefits_1[i] = netBenefit(y_true, y_prob_1, th)
    plt.figure()
    plt.plot(thresholds, net_benefits, color='blue', label='MODEL')
    plt.plot(thresholds, net_benefits_0, color='black', label='None')
    plt.plot(thresholds, net_benefits_1, color='red', label='All')
    plt.legend(loc='lower right')
    plt.title('DCA (' + dataset + ')')
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.ylim(-0.1, 1)
    plt.show()

def netBenefit(y_true, y_prob, threshold):
    y_pred = np.zeros(y_prob.shape)
    y_pred[np.where(y_prob > threshold)] = 1
    num = y_true.size
    tp_num = len(np.intersect1d(np.where(y_true == 1), np.where(y_pred == 1)))
    fp_num = len(np.intersect1d(np.where(y_true == 0), np.where(y_pred == 1)))

    tpr = tp_num / num
    fpr = fp_num / num

    NB = tpr - fpr * threshold / (1 - threshold)
    return NB

def plot_calibration_curve(est, y_test, prob_pos, name, bins=4, dataset=''):
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    # lr = LogisticRegression(C=1., solver='lbfgs')

    plt.figure()

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        '''
        # predict probability of positive class 
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        '''

        clf_score = metrics.brier_score_loss(y_test, prob_pos, pos_label=y_test.max())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=bins)

        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title('{} Calibration plots (reliability curve)'.format(dataset))
    plt.show()

def convert_df(patient_infos_per_dataset: list):
    df = pd.DataFrame()
    for dataset_id, patient_infos in enumerate(patient_infos_per_dataset):
        for patient_id in patient_infos:
            df.loc[patient_id, 'Label'] = patient_infos[patient_id]['gt_label']
            df.loc[patient_id, 'pred'] = patient_infos[patient_id]['pred'][1]
            df.loc[patient_id, 'dataset_id'] = dataset_id
    df.index.name = 'patient_id'
    return df

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # cfg.data.train['pipeline'] = cfg.data.test['pipeline']
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)
    val_dataset = build_dataset(cfg.data.val)
    train_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    test_data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    val_data_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    init_model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(init_model)
    checkpoint = load_checkpoint(init_model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    if not distributed:
        if args.device == 'cpu':
            model = init_model.cpu()
        else:
            model = MMDataParallel(init_model, device_ids=[0])
        model.CLASSES = CLASSES
        val_results = get_model_prob(model, val_data_loader)
        test_results = get_model_prob(model, test_data_loader)
        train_results = get_model_prob(model, train_data_loader)
    else:
        # val_results = None
        # test_results = None
        raise NotImplementedError
    val_pred_per_patient, val_gt_per_patient, val_patient_infos = \
        val_dataset.get_patient_labels_and_preds(val_results)
    test_pred_per_patient, test_gt_per_patient, test_patient_infos = \
        test_dataset.get_patient_labels_and_preds(test_results)
    train_pred_per_patient, train_gt_per_patient, train_patient_infos = train_dataset.get_patient_labels_and_preds(
        train_results)
    plot_figs(val_pred_per_patient, val_gt_per_patient, 'val')
    plot_figs(test_pred_per_patient, test_gt_per_patient, 'test')
    plot_figs(train_pred_per_patient, train_gt_per_patient, 'train')
    df = convert_df([train_patient_infos, val_patient_infos, test_patient_infos])
    if args is not None and not os.path.exists(os.path.abspath(os.path.dirname(args.out))):
        os.makedirs(os.path.abspath(os.path.dirname(args.out)))
    if args.out:
        df.to_csv(args.out)

if __name__ == '__main__':
    main()
