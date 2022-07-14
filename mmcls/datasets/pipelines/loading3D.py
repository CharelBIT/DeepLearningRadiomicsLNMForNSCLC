from ..builder import PIPELINES

import numpy as np
from ..builder import PIPELINES
import nibabel as nib
@PIPELINES.register_module()
class LoadImageFromNIIFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if isinstance(results["img_info"]['filename'], list) or \
            isinstance(results["img_info"]['filename'], tuple):
            filename = []
            img = []
            for img_path in results["img_info"]['filename']:
                filename.append(img_path)
                results['img_affine_matrix'] = []
                if "nii.gz" in img_path:
                    img_nii = nib.load(img_path)
                    # img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
                    results['img_affine_matrix'].append(img_nii.affine)
                    if len(img) != 0:
                        assert img_nii.shape == img[0].shape, "different mode must have same image shape"
                    img.append(img_nii)
                else:
                    print("[ERROR] Unspported 3D image format")
                    raise ValueError
            results['img_shape'] = img[0].shape
            results['ori_shape'] = img[0].shape
            results['pad_shape'] = img[0].shape
            results["num_mode"] = len(img)
        else:
            img_nii = nib.load(results["img_info"]['filename'])
            # img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
            img = img_nii
            filename = results['img_info']['filename']
            results['img_affine_matrix'] = img_nii.affine
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['pad_shape'] = img.shape
            results["num_mode"] = 1
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img

        results['scale_factor'] = 1.0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        # repr_str += f'(to_float32={self.to_float32},'
        # repr_str += f"color_type='{self.color_type}',"
        # repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsFromNIIFile(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self, reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label
        # self.ignore_black_area = ignore_black_area
        # self.file_client_args = file_client_args.copy()
        # self.file_client = None
        # self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        filename = results['ann_info']['seg_map']
        img_nii = nib.load(filename)
        results["seg_affine_matrix"] = img_nii.affine
        results['gt_semantic_seg'] = img_nii
        if 'seg_fields' not in results:
            results['seg_fields'] = []
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        return repr_str