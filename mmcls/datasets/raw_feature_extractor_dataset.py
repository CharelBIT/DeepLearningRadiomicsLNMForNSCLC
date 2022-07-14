import os
import numpy as np
from glob import glob
import imagesize
import os.path as osp
from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
@DATASETS.register_module()
class RawFeatureExtractorDataset(BaseDataset):
    CLASSES = ("background", "1")
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.bmp',
                 data_root=None,
                 test_mode=False,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 ignore_index=255):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
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
        # load annotations
        self.data_infos = self.load_annotations(self.img_dir)

    def load_annotations(self, img_dir):
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
        files = glob(os.path.join(img_dir, '*' + self.img_suffix))
        assert len(files) > 0
        for file in files:
            img_info = {}
            img_info['img_info'] = dict(file_name=file)
            w, h = imagesize.get(file)
            img_info['img_info']['width'] = w
            img_info['img_info']['height'] = h
            img_info['img_info']['filename'] = img_info['img_info']['file_name']
            img_info['img_prefix'] = img_dir
            # ATTENTION: FAKE Lable
            img_info['gt_label'] = np.array(1, dtype=np.int64)
            img_infos.append(img_info)
        return img_infos

    def evaluate(self,
               results,
               metric='accuracy',
               metric_options=None,
               logger=None):
        raise NotImplementedError
