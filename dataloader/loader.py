import os
import math
from random import sample

import peroneal
import peroneal_vit

def mktv(pth, tvs):
    """
    Args:
        pth (str)   path to data train/val/test txt file directory     
            Ex. ```/home/dongik/datasets/CPN/splits/v5/3```
                ```/home/dongik/datasets/CPN/splits```
        tvs (int)   train/validate dataset ratio
            Ex. 2 block     ```1 mini-block train set, 1 mini-block validate set```
                5 block     ```4 mini-block train set, 1 mini-block validate set```
    """
    if not os.path.exists( os.path.join(pth, '_train.txt') ):
        raise Exception( '_train.txt not found. ', os.path.join(pth, '_train.txt') )
    
    with open(os.path.join(pth, '_train.txt'), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    n = math.floor(len(file_names) / tvs)

    val = sample(file_names, n)
    train = list(set(file_names) - set(val))

    with open(os.path.join(pth, 'train.txt'), 'w') as f:
        for w in train:
            f.write(f'{w}\n')
    with open(os.path.join(pth, 'val.txt'), 'w') as f:
        for w in val:
            f.write(f'{w}\n')

def cpn(pth, tvs, mkset = False, **kwargs):
    
    """ -Peroneal nerve ( 490 samples )
        fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP (0 ~ 5)
    Args:
        mkset (bool)   :  True for make new train/val set
    """
    if tvs < 2 and mkset:
        raise Exception("tvs must be larger than 1")
    elif tvs >= 2 and mkset:
        mktv(pth, tvs)

    return peroneal.Peroneal(**kwargs)

def cpn_vit(pth, tvs, mkset = False, **kwargs):
    
    """ -Peroneal nerve ViT dataset ( 1648 samples )
    Args:
        mkset (bool)   :  True for make new train/val set
    """
    if tvs < 2 and mkset:
        raise Exception("tvs must be larger than 1")
    elif tvs >= 2 and mkset:
        mktv(pth, tvs)

    return peroneal_vit.PeronealViT(**kwargs)




if __name__ == "__main__":
    import sys

    print(f'file abs path: {os.path.abspath(__file__)}')
    print(f'dirname: {os.path.dirname( os.path.abspath(__file__) )}')

    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader

    transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dst = cpn_vit(pth='/home/dongik/datasets/CPN_vit/splits/v5/3', tvs=5, mkset=True,
                    root='/home/dongik/datasets', dver='splits/v5/3', image_set='train', 
                    transform=transform, in_channels=3,  image_patch_size=(64, 64))

    loader = DataLoader(dst, batch_size=16,
                        shuffle=True, num_workers=4, drop_last=True)
    
    print(f'[train] {len(dst)} samples')
    for i, (ims, lbls) in (enumerate(loader)):
        print(f'image shape: {ims.shape}')
        print(f'label shape: {lbls[1].shape}')
        print(f'vit lbl: {lbls[0].shape}')
        print(f'ROI: {lbls[1].numpy().sum()/(lbls[1].shape[0] * lbls[1].shape[1] * lbls[1].shape[2]):.4f}')
        print(f'BG: {1 - lbls[1].numpy().sum()/(lbls[1].shape[0] * lbls[1].shape[1] * lbls[1].shape[2]):.4f}')
        if i > 0:
            break