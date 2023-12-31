import numpy as np
import sys
sys.path.append('..')
from utils import project_root
from utils.serialization import json_load
from dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'advent/dataset/Paris/info.json'


class Paris(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(512, 512), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)

    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
    """def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        #label = self.get_labels(label_file)
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), image.copy(), np.array(image.shape), name"""