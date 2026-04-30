




from __future__ import annotations
from typing import Any, Dict
import glob
import os
import os.path as osp
import json
import numpy as np
import torch
import random
from PIL import Image
from dataset.utils import split_selector, get_transform

class MultiByPassT40(torch.utils.data.Dataset):
    def __init__(self,
            config: Dict[str, Any],
            test_fold: int = 1,
            video_dir_prefix: str = 'videos',
            video_path: str = 'MultiBypassT40/videos',
            label_path: str = 'MultiBypassT40/labels',
            img_size: tuple[int, int] = (224, 224),
            split: str = 'train',
            aug = None):
        self.config = config
        self.split = split.lower()
        self.setting = config.dataset.setting.lower()
        self.test_fold = test_fold
        self.video_dir_prefix = video_dir_prefix
        self.video_path = osp.join(self.video_dir_prefix, video_path)
        self.label_path = osp.join(self.video_dir_prefix, label_path)
        self.img_size = img_size
        self.sampling_percentage = config.dataset.sampling_percentage
        self.clip_len = config.dataset.clip_len
        self.clip_position = config.dataset.clip_position
        self.clip_center_mode = config.dataset.clip_center_mode
        self.aug = aug
        self.clip_consistent_augs = bool(getattr(config.dataset, "clip_consistent_augs", True))
        self._load_data()
        train_transform, test_transform = get_transform(config, img_size, aug)

        if 'train' in self.split:
            self.transform = train_transform
        else:
            self.transform = test_transform

    def _get_clip_frame_keys(self, video_id: str, frame_key: int):
        if self.clip_len == 1:
            return [os.path.join(self.video_path, video_id, f'{frame_key:06d}.jpg')]
        else:
            # buiding video with frame keys in a causal style
            start_frame = max(frame_key - self.clip_len + 1, 0)
            clip_frame_keys = list(range(start_frame, frame_key + 1))

            if len(clip_frame_keys) < self.clip_len:
                pad_count = self.clip_len - len(clip_frame_keys)
                pad_value = clip_frame_keys[0] if clip_frame_keys else frame_key
                clip_frame_keys = [pad_value] * pad_count + clip_frame_keys

            # add path prefix to the frame keys
            clip_frame_keys = [os.path.join(self.video_path, video_id, f'{frame_id:06d}.jpg') for frame_id in clip_frame_keys]
            return clip_frame_keys

    def _create_zero_labels(self, 
                            num_classes_i, 
                            num_classes_v, 
                            num_classes_t, 
                            num_classes_ivt):
        i_labels = torch.zeros(num_classes_i)
        v_labels = torch.zeros(num_classes_v)
        t_labels = torch.zeros(num_classes_t)
        ivt_labels = torch.zeros(num_classes_ivt)
        return i_labels, v_labels, t_labels, ivt_labels

    def _load_data(self):
        label_files = glob.glob(os.path.join(self.label_path, '*.json'))
        video_records = set(split_selector(dataset_name='multibypasst40', setting=self.setting, split=self.split, test_fold=self.test_fold))
        data_list = []
        self.video_distribution = {}
        num_classes_i = self.config['model']['num_tool_classes']
        num_classes_v = self.config['model']['num_verb_classes']
        num_classes_t = self.config['model']['num_target_classes']
        num_classes_ivt = self.config['model']['num_triplet_classes']
        i_list, v_list, t_list, ivt_list = [], [], [], []

        for label_file in label_files:
            video_id = os.path.splitext(os.path.basename(label_file))[0]            
            if video_id not in video_records:
                continue

            with open(label_file, 'r') as f:
                data = json.load(f)
            self.video_distribution[video_id] = len(data['images'])

            video_frame_list = data['images']
            video_frame_annotations = data['annotations']
            ann_keys = sorted([fi['id'] for fi in video_frame_list])
            
            for frame_info in video_frame_list:
                frame_key = frame_info['id']
                frame_filename = frame_info['file_name']

                # get all annotations for the current frame
                frame_anns = []
                for ann in video_frame_annotations:
                    if ann['image_id'] == frame_key:
                        frame_anns.append(ann)

                label_i, label_v, label_t, label_ivt = self._create_zero_labels(
                    num_classes_i, 
                    num_classes_v, 
                    num_classes_t,
                    num_classes_ivt
                )
                if len(frame_anns) > 0:
                    for ann in frame_anns:
                        triplet_cls_id = ann['category_id']
                        instrument_cls_id = ann['instrument_id']
                        verb_cls_id = ann['verb_id']
                        target_cls_id = ann['target_id']
                        label_i[instrument_cls_id] = 1
                        label_v[verb_cls_id] = 1
                        label_t[target_cls_id] = 1
                        label_ivt[triplet_cls_id] = 1

                ivt_list.append(label_ivt.tolist())
                i_list.append(label_i.tolist())
                v_list.append(label_v.tolist())
                t_list.append(label_t.tolist())
                clip_frame_keys = self._get_clip_frame_keys(video_id, frame_key)

                data_list.append({
                    'label_i': label_i,
                    'label_v': label_v,
                    'label_t': label_t,
                    'label_ivt': label_ivt,
                    'video_id': video_id,
                    'frame_key': torch.tensor(frame_key),
                    'filename': os.path.join(self.video_path, video_id, frame_filename),
                    'filename_clip': clip_frame_keys,
                })
            
        self.data = data_list

        if self.sampling_percentage < 1.0 and self.split == 'train':
            self.data = self.data[:int(len(self.data) * self.sampling_percentage)]
        
        # Get the class frequencies of all classes in the ivt_list
        class_counts = np.sum(np.array(ivt_list), axis=0)
        total_instances = len(data_list)
        num_zero_classes = np.where(class_counts == 0)[0].shape[0] # Print dataset statistics
        print(f"Split: {self.split} | Zero count classes: {np.where(class_counts == 0)[0].tolist()}")
        print(f"Total valid classes: {num_classes_ivt - num_zero_classes}/{num_classes_ivt}")
        print(f"Total samples: {total_instances}")

    def __len__(self):
        return len(self.data)

    def _transform_images_clip_consistent(self, images):
        """Apply the same stochastic transform parameters to every frame in a clip."""
        torch_rng_state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()
        transformed_images = []
        for image in images:
            torch.set_rng_state(torch_rng_state)
            random.setstate(random_state)
            np.random.set_state(np_state)
            transformed_images.append(self.transform(image))
        return transformed_images

    def _transform_images_per_frame(self, images):
        """Legacy behavior: each frame gets independently sampled augmentation params."""
        return [self.transform(image) for image in images]

    def __getitem__(self, idx):
        item = self.data[idx]
        images = [Image.open(filename).convert('RGB') for filename in item['filename_clip']]
        if self.transform is not None:
            if self.clip_consistent_augs:
                transformed_images = self._transform_images_clip_consistent(images)
            else:
                transformed_images = self._transform_images_per_frame(images)
        else:
            transformed_images = images

        tensor = torch.stack(transformed_images) if isinstance(transformed_images[0], torch.Tensor) else transformed_images
        item_out = dict(item)
        item_out['image'] = tensor
        return item_out

