from torch.utils.data import Dataset
import torch as t
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    
    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode # Values: val or train
        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize(train_mean, train_std)
        ])
        # TODO: Consider creating two different
        # transforms based on whether you are in the training or validation dataset.
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        fname = self.data['filename'].iloc[index] # Get image name
        image = imread('images/'+fname) # Read image
        labels = self.data[['crack', 'inactive']].iloc[index] # Get labels
        rgb_img = gray2rgb(image) # Covnert to rgb
        output_image = self._transform(rgb_img)

        return output_image, t.as_tensor(labels)