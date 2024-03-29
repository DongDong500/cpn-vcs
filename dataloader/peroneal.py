import os
import sys
import math
import numpy as np
import torch.utils.data as data
from random import sample
from PIL import Image

class Peroneal(data.Dataset):
    """
    Args:6
        root (string): Root directory of the Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train``, ``val`` or ``test``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        dver (str): version of dataset (ex) ``splits/v5/3``
    """
    def get_lbl(self, target):
        size = target.size #(width, height)
        mask = np.array(target, dtype=np.uint8)
        h, w = np.where(mask > 0)
        tl = (h.min(), w.min())
        rb = (h.max(), w.max())
        pnt = ( int((tl[0] + rb[0])/2), int((tl[1] + rb[1])/2) )
        # clsn = 4 * ((pnt[0] // 128)) + ((pnt[1] // 128))
        clsn = ( pnt[0] // self.image_size_h, pnt[1] // self.image_size_w )

        #return np.expand_dims(np.array(clsn, dtype=np.float32), axis=0) / 1024
        return (size[0] / self.image_size_w) * clsn[0] + clsn[1]

    def read(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError
        
        if self.in_channels == 3:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')         
        else:
            raise Exception ("in channel must be 3")

        assert( img.size == target.size == (512, 512) )

        target = (self.get_lbl(target), target)

        return img, target

    def __init__(self, root, datatype='cpn', dver='splits', image_set='train', 
                    transform=None, in_channels=3, image_patch_size=(64, 64), **kwargs):
        self.root = root
        self.datatype = datatype
        self.dver = dver
        self.image_set = image_set
        self.transform = transform
        self.in_channels = in_channels
        self.image_size_h = image_patch_size[0]
        self.image_size_w  = image_patch_size[1]

        image_dir = os.path.join(self.root, self.datatype, 'Images')
        mask_dir = os.path.join(self.root, self.datatype, 'Masks')
        
        if self.image_set == 'train' or self.image_set == 'val':
            split_f = os.path.join(self.root, self.datatype, self.dver, 
                                    kwargs["cur_time"], kwargs["rid"], self.image_set.rstrip('\n') + '.txt')
        else:
            split_f = os.path.join(self.root, self.datatype, self.dver, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise Exception('Dataset not found or corrupted.')
    
        if not os.path.exists(split_f):
            raise Exception('Wrong image_set entered!' 
                            'Please use image_set="train" or image_set="val"\n', split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".bmp") for x in file_names]
        
        assert (len(self.images) == len(self.masks))

        self.image = []
        self.mask = []
        for index in range(len(self.images)):
            img, tar = self.read(index)
            self.image.append(img)
            self.mask.append(tar)

    def __getitem__(self, index):

        img = self.image[index]
        target = self.mask[index][1]
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, (self.mask[index][0], target)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    image_set_type = ['train', 'val', 'test']
    for ist in image_set_type:
        dst = Peroneal(root='/home/dongik/datasets', datatype='cpn', image_set=ist,
                    transform=transform, in_channels=3, dver='splits/v5/3', tvs=20)
        loader = DataLoader(dst, batch_size=16,
                                shuffle=True, num_workers=2, drop_last=True)
        print(f'[{ist}] {len(dst)} samples')

        for i, (ims, lbls) in enumerate(loader):
            if i < 1:
                print(f'ims shape {ims.shape}')
                print(f'lbls shape {lbls[1].shape}')
            pass
        
        print('Clear !!!')