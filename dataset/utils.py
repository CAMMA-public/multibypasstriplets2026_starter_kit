from typing import Any, Dict
from torchvision import transforms as T
from randaugment import RandAugment

def split_selector(dataset_name='cholect50', setting='cholect50', split='train', test_fold=1):
    switcher = {
        'cholect50': {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 
                35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val'  : [8, 12, 29, 50, 78],
                'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 
                13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val':   [5, 18, 36, 65, 74],
                'test':  [92, 96, 103, 110, 111]
            },
            'cholect45-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
            },
            'cholect50-crossval': {
                1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
                2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
                3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
            },
        },
        'multibypasst40': {
                # NOTE: below are the released video frames for the challenge. This is one custom split provided for quick development and testing and is not the final
                # split. Participants are free to create their own splits for training and validation, they are eventually evaluated on the test set and 
                # the hidden test set that are with the challenge organizers.
                "challenge" :{
                    'train': ['C1V1', 'C1V3', 'C1V4', 'C1V5', 'C1V6', 'C1V7', 'C2V1', 'C2V10', 'C2V11', 'C2V12', 'C2V14', 'C2V2', 'C2V3', 'C2V4'],
                    'val': ['C2V5'],
                    'test': ['C2V6']
                },
        },
    }
    
    video_split  = switcher.get(dataset_name.lower()).get(setting.lower())
    if dataset_name == 'cholect50':
        train_videos = sum([v for k,v in video_split.items() if k!=test_fold], []) if 'crossval' in setting else video_split['train']
        test_videos  = sum([v for k,v in video_split.items() if k==test_fold], []) if 'crossval' in setting else video_split['test']
        if 'crossval' in setting:
            val_videos   = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos   = video_split['val']
        train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        val_records   = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        test_records  = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
    else:
        if 'crossval' in setting:
            train_records = video_split[test_fold]['train']
            val_records = video_split[test_fold]['val']
            test_records = video_split[test_fold]['test']
        else:
            train_records = video_split['train']
            val_records = video_split['val']
            test_records = video_split['test']
            if 'hidden_test' in video_split:
                hidden_test_records = video_split['hidden_test']

    if split == 'train':
        return train_records
    elif split == 'val':
        return val_records
    elif split == 'test':
        return test_records
    elif split == 'hidden_test':
        return hidden_test_records
    else:
        raise ValueError(f"Invalid split: {split}")

def get_transform(config: Dict[str, Any], img_size: tuple[int, int], aug: str) -> T.Compose:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if config.dataset.aug_type == "aug0":
            train_transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        elif config.dataset.aug_type == "aug1":
            train_transform = T.Compose([
                T.Resize(img_size),
                RandAugment(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            train_transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
                
        test_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        return train_transform, test_transform
