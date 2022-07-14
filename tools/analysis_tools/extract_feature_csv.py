import argparse
import os
import warnings
from tqdm import tqdm
import mmcv
import torch
import pandas as pd
from mmcv import DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

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
    parser.add_argument('--save-path', help='checkpoint file')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
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
    parser.add_argument('--gpu-idx', default=0, type=int, help="device used for testing")
    parser.add_argument('--feature-name', default="", type=str, help='feature name for exp')
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
    if hasattr(cfg.data, 'test'):
        cfg.data.test.test_mode = True

    try:
        cfg.data.train.pipeline = cfg.feature_extract_pipeline
        if hasattr(cfg.data, 'test'):
            cfg.data.test.pipeline = cfg.feature_extract_pipeline
    except:
        print("[WARNING] 'feature_extract_pipline' not in cfg file "
              "using test_pipline")
        for operation in cfg.test_pipeline:
            if operation['type'] == 'ToTensor' or operation['type'] == 'Collect':
                operation['keys'] = ['img', 'gt_label']
        cfg.data.train.pipeline = cfg.test_pipeline
        if hasattr(cfg.data, 'test'):
            cfg.data.test.pipeline = cfg.test_pipeline

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = None
    if hasattr(cfg.data, 'test'):
        test_dataset = build_dataset(cfg.data.test)
    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    results = []
    model = model.to("cuda:{}".format(args.gpu_idx))
    pbar = tqdm(train_dataset)
    test_count = 1
    for i, d in enumerate(pbar):
        # if i > test_count:
        #     break
        pbar.set_description("Train Dataset")
        with torch.no_grad():
            feats = model.extract_feat(d['img'][None, ...].to("cuda:{}".format(args.gpu_idx)))
            result = dict(file_name=d['img_metas'].data['filename'],
                          features=feats,
                          gt_label=d['gt_label'].item(),
                          train=1)
            results.append(result)
    if test_dataset is not None:
        pbar = tqdm(test_dataset)
        for i, d in enumerate(pbar):
            # if i > test_count:
            #     break
            pbar.set_description("Test Dataset")
            with torch.no_grad():
                feats = model.extract_feat(d['img'][None, ...].to("cuda:{}".format(args.gpu_idx)))
                result = dict(file_name=d['img_metas'].data['filename'],
                              features=feats,
                              gt_label=d['gt_label'].cpu().item(),
                              train=0)
                results.append(result)
    df = pd.DataFrame()
    for i, result in enumerate(results):
        feats = []
        if isinstance(result['features'], torch.Tensor):
            feats.extend(result['features'][0].cpu().numpy().tolist())
        else:
            for f in result['features']:
                feats.extend(f[0].cpu().numpy().tolist())
        df.loc[i, 'file_name'] = result['file_name']
        df.loc[i, 'train_sample'] = int(result['train'])
        df.loc[i, 'Label'] = int(result['gt_label'])
        for j, f in enumerate(feats):
            df.loc[i, args.feature_name + '_' + str(j)] = f
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    df.to_csv('{}/{}.csv'.format(args.save_path, args.config.split('/')[-1][:-3]))
if __name__ == '__main__':
    main()
