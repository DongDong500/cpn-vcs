import os
import sys
import math
from random import sample

try:
    # from .cpn import CPN
    from .cpn_vit import CPNvit
    from .median import Median
    from utils.ext_transforms import ExtCompose
except:
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    # from cpn import CPN
    from cpn_vit import CPNvit
    from median import Median
    from utils.ext_transforms import ExtCompose


def mktv(root:str = '/', datatype:str = 'CPN', dver:str = 'splits/v5/3', tvs:int = 5):
    split_f = os.path.join(root, datatype, dver, '_train.txt')

    if not os.path.exists(split_f):
        raise Exception(f'_train.txt not found or corrupted. {split_f}')
    
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    n = math.floor(len(file_names) / tvs)

    val = sample(file_names, n)
    train = list(set(file_names) - set(val))

    with open(os.path.join(root, datatype, dver, 'train.txt'), 'w') as f:
        for w in train:
            f.write(f'{w}\n')
    with open(os.path.join(root, datatype, dver, 'val.txt'), 'w') as f:
        for w in val:
            f.write(f'{w}\n')
    
def cpn(root:str = '/', datatype:str = 'CPN', dver:str = 'splits/v5/3',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
    Args:
        root (str)  :   path to data parent directory (Ex: /data1/sdi/datasets) 
        datatype (str)  :   data folder name (default: CPN_all)
        dver (str)  : version of dataset (default: splits)
        image_set (str) :    train/val or test (default: train)
        transform (ExtCompose)  :   composition of transform class
        is_rgb (bool)   :  True for RGB, False for gray scale images
        tvs (int)   :  train/validate dataset ratio 
                2 block = 1 mini-block train set, 1 mini-block validate set
                5 block = 4 mini-block train set, 1 mini-block validate set
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN', dver, tvs)

    return CPN(root, 'CPN' ,dver, image_set, transform, is_rgb)

def median(root:str = '/', datatype:str = 'Median', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ - Median nerve:     1305 samples (1044 + 261)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'Median', 'splits', tvs)

    return Median(root, 'Median', dver, image_set, transform, is_rgb)



if __name__ == "__main__":
    print(f'file abs path: {os.path.abspath(__file__)}')
    print(f'dirname: {os.path.dirname( os.path.abspath(__file__) )}')

    #sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), is_crop=True, pad_if_needed=True),
            et.ExtScale(scale=0.5, is_scale=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dst = cpn(root='/home/dongik/datasets', dver='splits/v5/3',
                image_set='train', transform=transform, is_rgb=True, tvs=5)

    loader = DataLoader(dst, batch_size=16,
                        shuffle=True, num_workers=4, drop_last=True)
    print(f'dataset len(dst) = {len(dst)}')
    for i, (ims, lbls) in (enumerate(loader)):
        print(f'image shape: {ims.shape}')
        print(f'label shape: {lbls.shape}')
        print(f'ROI: {lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]):.4f}')
        print(f'BG: {1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]):.4f}')
        if i > 0:
            break