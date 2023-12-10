
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from ultralytics.models.yolo.detect import DetectionTrainer
import utils
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset
import torch
from PIL import Image
from ultralytics.data import YOLODataset
from ultralytics.data.utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable

import os
import numpy as np


# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'


class CustomDataset(YOLODataset):


    def __init__(self, df, label_to_id, img_path, data=None, *args, **kwargs):
        super().__init__(img_path, data=data, *args, **kwargs)
        self.df = df
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        print(img_path)



    def __len__(self):
        return len(self.df)

    """ 
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image_path = data.image

        print("data: ", data)

        if self.image_path is not None:
            image_path = self.image_path / image_path

        raw_image = Image.open(image_path)
        image = raw_image.convert("RGB")

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        raw_label = data.label
        label = self.label_to_id[raw_label]

        return image, label 
    
    """



    def get_labels(self):
        
        """ Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        self.label_files = img2label_paths(self.im_files)
        print("label_files: ", self.label_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files
        print("im_files: ",  self.im_files)
        
        return labels


        
        """ 
        labels_dict = {}
        data = self.df.iloc[0]
        image_path = data.image

        print("data: ", data)

        if self.image_path is not None:
            image_path = self.image_path / image_path

        raw_image = Image.open(image_path)
        image = raw_image.convert("RGB")

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        raw_label = data.label
        label = self.label_to_id[raw_label]

        return image, label
        """

    

class CustomTrainer(DetectionTrainer):
    """ Custom trainer for imbalanced data """

    def get_dataloader(self, data, batch_size=16, rank=0, mode='train'):
        train_images_path = Path(data).parent/'images'
        #print("train_images_path: ", train_images_path)
        
        imbalanced_train_df = create_df(train_images_path)
        #print("imbalanced_train_df: \n", imbalanced_train_df.head())
        #print('Train distribution')
        #print(imbalanced_train_df.labels.value_counts())
        #print('==============')
        #print('Validation distribution')
        #print(val_df.labels.value_counts())

        label_to_id = {'bifurcation': 0, 'background': 1}
        num_classes = len(label_to_id)

        #print("labels: ", label_to_id)

        class_counts = imbalanced_train_df.label.value_counts()
        class_weights = 1/class_counts
        sample_weights = [1/class_counts[i] for i in imbalanced_train_df.label.values]
        #print(class_weights)

        #ds = TensorDataset(torch.as_tensor([(idx, label_to_id[l]) for idx, l in enumerate(imbalanced_train_df.label.values)]))
        train_dataset = CustomDataset(imbalanced_train_df, label_to_id, img_path='/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_2/train/images', data=data, augment=False)

        sampler= WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

        return dataloader

def create_df(train_images_path):
    train_label_path = train_images_path.parent/'labels'
    val_data_path = train_images_path.parent.parent/'val/images'
    val_label_path = val_data_path.parent/'labels'
    train_images, train_labels, bbox = [], [], []
    for lbl in train_label_path.glob('*.txt'):
        coord = utils.read_label_txt(train_label_path/lbl)
        if len(coord) != 0:
            train_labels.append('bifurcation')
            bbox.append(coord)
        else:
            train_labels.append('background')
            bbox.append('')
        train_images.append('_'.join(lbl.parts[-1].split('.')[0:-1]))
    imbalanced_train_df = pd.DataFrame(data={'image': train_images, 'label': train_labels, 'bbox': bbox})
    return imbalanced_train_df





def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache




if __name__ == '__main__':
    dataset_path = '/Users/aibotasanatbek/Documents/projects/calcium_scoring/data/datasets/train_val/split_2/split_2_dataset.yaml'
    args = dict(model='yolov8n.pt', data=dataset_path, epochs=30, imgsz=512, exist_ok=True)
    trainer = CustomTrainer(overrides=args)
    trainer.train()



    """
    train_images_path = Path(dataset_path).parent/'train/images'
    print("train_images_path: ", train_images_path)
    output_folder = Path(dataset_path).parent/'output'
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    imbalanced_train_df = create_df(train_images_path)
    print("imbalanced_train_df: \n", imbalanced_train_df.head())
    print('Train distribution')
    print(imbalanced_train_df.label.value_counts())
    print('==============')
    print('Validation distribution')
    #print(val_df.labels.value_counts())

    label_to_id = {'bifurcation': 0, 'background': 1}
    num_classes = len(label_to_id)

    class_counts = imbalanced_train_df.label.value_counts()
    class_weights = 1/class_counts
    sample_weights = [1/class_counts[i] for i in imbalanced_train_df.label.values]

    #
    # train_dataset = CustomDataset(df=imbalanced_train_df, label_to_id=label_to_id)
    ds = TensorDataset(torch.as_tensor([(idx, label_to_id[l]) for idx, l in enumerate(imbalanced_train_df.label.values)]))

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(ds), replacement=True)
    dataloader = DataLoader(ds, sampler=sampler, batch_size=32)


    for element in dataloader:
        print(element)


    """

