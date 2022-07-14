import argparse
import os
import warnings
import pandas as pd
import mmcv
from tqdm import tqdm
import numpy as np
import glob
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from sklearn.metrics import average_precision_score
# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('workdir', help='workdir')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save-result', default=None, type=str, help='save result per checkpoint')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
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


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # cfg.data.train['pipeline'] = cfg.data.test['pipeline']

    if isinstance(cfg.data.test, (list, tuple)):
        test_dataset = []
        if len(cfg.data.test) == 0:
            cfg.data.train['pipeline'] = cfg.test_pipeline
        else:
            cfg.data.train['pipeline'] = cfg.data.test[0]['pipeline']
        for i in range(len(cfg.data.test)):
            cfg.data.test[i].test_mode = True
            test_dataset.append(build_dataset(cfg.data.test[i]))
    else:
        cfg.data.test.test_mode = True
        cfg.data.train['pipeline'] = cfg.data.test['pipeline']
        test_dataset = build_dataset(cfg.data.test)
    val_dataset = build_dataset(cfg.data.val)
    # the extra round_up data will be removed during gpu/cpu collect
    train_dataset = build_dataset(cfg.data.train)
    train_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    if isinstance(cfg.data.test, (list, tuple)):
        test_data_loader = []
        for i in range(len(test_dataset)):
            test_data_loader.append(build_dataloader(
                test_dataset[i],
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
                round_up=True))
    else:
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

    # build the model and load checkpoint
    init_model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(init_model)
    # model_files = os.listdir(args.workdir)
    model_files = glob.glob(os.path.join(args.workdir, '*.pth'))
    result_df = pd.DataFrame()
    for model_file in tqdm(model_files):
        checkpoint = load_checkpoint(init_model, model_file, map_location='cpu')
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
            show_kwargs = {} if args.show_options is None else args.show_options
            if not isinstance(test_data_loader, list):
                test_outputs = single_gpu_test(model, test_data_loader, args.show, args.show_dir,
                                          **show_kwargs)
            else:
                test_outputs = []
                for i in range(len(test_data_loader)):
                    test_outputs.append(
                        single_gpu_test(model, test_data_loader[i], args.show, args.show_dir,
                                        **show_kwargs)
                    )
            val_outputs = single_gpu_test(model, val_data_loader, args.show, args.show_dir,
                                           **show_kwargs)
            train_outputs = single_gpu_test(model, train_data_loader, args.show, args.show_dir,
                                          **show_kwargs)
        else:
            raise NotImplementedError
            # model = MMDistributedDataParallel(
            #     init_model.cuda(),
            #     device_ids=[torch.cuda.current_device()],
            #     broadcast_buffers=False)
            # test_outputs = multi_gpu_test(model, test_data_loader, args.tmpdir,
            #                          args.gpu_collect)
            # val_outputs = multi_gpu_test(model, val_data_loader, args.tmpdir,
            #                               args.gpu_collect)
            # train_outputs = multi_gpu_test(model, train_data_loader, args.tmpdir,
            #                              args.gpu_collect)
        print("[INFO] metrics {}: ".format(model_file))
        rank, _ = get_dist_info()
        if rank == 0:
            results = {}
            if args.metrics:
                if not isinstance(test_dataset, list):
                    test_eval_results = test_dataset.evaluate(test_outputs, args.metrics,
                                                    args.metric_options)
                else:
                    test_eval_results = []
                    for i in range(len(test_dataset)):
                        test_eval_results.append(
                            test_dataset[i].evaluate(test_outputs[i], args.metrics,
                                                  args.metric_options)
                        )
                val_eval_results = val_dataset.evaluate(val_outputs, args.metrics,
                                                          args.metric_options)
                train_eval_results = train_dataset.evaluate(train_outputs, args.metrics,
                                                        args.metric_options)


                if not isinstance(test_eval_results, list):
                    keys = list(test_eval_results.keys())
                    for key in keys:
                        test_eval_results['test_' + key] = test_eval_results[key]
                        del test_eval_results[key]
                else:
                    for i in range(len(test_eval_results)):
                        keys = list(test_eval_results[i].keys())
                        for key in keys:
                            test_eval_results[i][f'test_{i}' + key] = test_eval_results[i][key]
                            del test_eval_results[i][key]

                keys = list(val_eval_results.keys())
                for key in keys:
                    val_eval_results['val_' + key] = val_eval_results[key]
                    del val_eval_results[key]

                keys = list(train_eval_results.keys())
                for key in keys:
                    train_eval_results['train_' + key] = train_eval_results[key]
                    del train_eval_results[key]
                if not isinstance(test_eval_results, list):
                    results.update(test_eval_results)
                else:
                    for i in range(len(test_eval_results)):
                        results.update(test_eval_results[i])
                results.update(val_eval_results)
                if not isinstance(test_eval_results, list):
                    test_eval_results = [test_eval_results]
                for i in range(len(test_eval_results)):
                    for k, v in test_eval_results[i].items():
                        try:
                            result_df.loc[model_file.split('/')[-1].split('.')[0], k] = v
                            print(f'\n{k} : {v:.2f}')
                        except:
                            continue
                for k, v in val_eval_results.items():
                    try:
                        result_df.loc[model_file.split('/')[-1].split('.')[0], k] = v
                        print(f'\n{k} : {v:.2f}')
                    except:
                        continue

                for k, v in train_eval_results.items():
                    try:
                        result_df.loc[model_file.split('/')[-1].split('.')[0], k] = v
                        print(f'\n{k} : {v:.2f}')
                    except:
                        continue
        if args.save_result is not None:
            result_df.to_csv(args.save_result)


if __name__ == '__main__':
    main()
