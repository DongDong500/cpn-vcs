import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter

from dataloader import Peroneal
from utils import ext_transforms as et

def get_dataset(opts, dataset, dver):
    mean = [0.485, 0.456, 0.406] if (opts.in_channels == 3) else [0.485]
    std = [0.229, 0.224, 0.225] if (opts.in_channels == 3) else [0.229]

    train_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        et.GaussianPerturb(mean=opts.mu, std=opts.std)
        ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        et.GaussianPerturb(mean=opts.mu_val, std=opts.std_val)
        ])
    test_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        et.GaussianPerturb(mean=opts.mu_test, std=opts.std_test)
        ])
    
    train_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='train', 
                                                    transform=train_transform, 
                                                    is_rgb=(opts.in_channels == 3), 
                                                    tvs=opts.tvs,
                                                    mu=opts.c_mu,
                                                    std=opts.c_std
                                                    )

    val_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='val', 
                                                    transform=val_transform, 
                                                    is_rgb=(opts.in_channels == 3), 
                                                    tvs=opts.tvs,
                                                    mu=opts.c_mu,
                                                    std=opts.c_std
                                                    )

    test_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='test', 
                                                    transform=test_transform, 
                                                    is_rgb=(opts.in_channels == 3))


def experiments(opts, run_id) -> dict:

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (devices, opts.gpus))

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    RUN_ID = 'run_' + str(run_id).zfill(2)
    os.mkdir(os.path.join(opts.Tlog_dir, RUN_ID))
    os.mkdir(os.path.join(opts.best_ckpt, RUN_ID))
    os.mkdir(os.path.join(opts.test_results_dir, RUN_ID))
    writer = SummaryWriter(log_dir=os.path.join(opts.Tlog_dir, RUN_ID))

