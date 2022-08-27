import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
import network
import criterion
import dataloader
from utils import ext_transforms as et

def get_dataset(opts):
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
        
    train_dst = dataloader.loader.__dict__[opts.dataset]( 
        pth = os.path.join(opts.data_root, opts.dataset, opts.dataset_ver), tvs = opts.tvs, mkset = True,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='train', transform=train_transform, in_channels=opts.in_channels )

    val_dst = dataloader.loader.__dict__[opts.dataset]( 
        pth = os.path.join(opts.data_root, opts.dataset, opts.dataset_ver), tvs = opts.tvs, mkset = False,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='val', transform=train_transform, in_channels=opts.in_channels )

    test_dst = dataloader.loader.__dict__[opts.dataset]( 
        pth = os.path.join(opts.data_root, opts.dataset, opts.dataset_ver), tvs = opts.tvs, mkset = False,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='test', transform=train_transform, in_channels=opts.in_channels)

    print("Dataset - %s\n\tTrain\t%d\n\tVal\t%d\n\tTest\t%d" % 
            (opts.dataset_ver + '/' + opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_dst, val_dst, test_dst

def load_model(opts, verbose = False):

    print(f'Seg model - {opts.model}') if verbose else 0
    print(f'ViT model - {opts.vit_model}') if verbose else 0

    if opts.model.startswith("deeplabv3plus"):
        segmodel = network.model.__dict__[opts.model](
            in_channels=opts.in_channels, classes=opts.classes,
            encoder_name=opts.encoder_name, encoder_depth=opts.encoder_depth,
            encoder_weights=opts.encoder_weights, encoder_output_stride=opts.encoder_output_stride,
            decoder_atrous_rates=opts.decoder_atrous_rates, decoder_channels=opts.decoder_channels,
            activation=opts.activation, upsampling=opts.upsampling, aux_params=opts.aux_params)
    else:
        raise NotImplementedError

    if opts.vit_model == "vit":
        vitmodel = network.model.__dict__[opts.vit_model](
            image_size=opts.vit_image_size, patch_size=opts.vit_patch_size, num_classes=opts.vit_num_classes,
            dim=opts.vit_dim, depth=opts.vit_depth, heads=opts.vit_heads, 
            mlp_dim=opts.vit_mlp_dim, dropout=opts.vit_dropout, emb_dropout=opts.vit_emb_dropout 
        )
    else:
        raise NotImplementedError

    if opts.model_pretrain and os.path.isfile(opts.model_params):
        print("restored parameters from %s" % opts.model_params) if verbose else 0
        ckpt = torch.load(opts.model_params, map_location=torch.device('cpu'))
        segmodel.load_state_dict( ckpt["model_state"] )

    if opts.model_pretrain and os.path.isfile(opts.vit_model_params):
        print("restored parameters from %s" % opts.vit_model_params) if verbose else 0
        ckpt = torch.load(opts.vit_model_params, map_location=torch.device('cpu'))
        vitmodel.load_state_dict( ckpt["model_state"] )
    
    del ckpt
    torch.cuda.empty_cache()

    return segmodel, vitmodel

def set_optim(opts, model_name, model):

    if model_name.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': model.encoder.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.decoder.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=opts.lr, 
            weight_decay=opts.weight_decay, momentum=opts.momentum)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    return optimizer, scheduler

def train(devices, model, loader, loss_type, optimizer, scheduler, metrics, **kwargs):

    model.train()
    metrics.reset()
    running_loss = 0.0
    lossFun = criterion.get_criterion.__dict__[loss_type](**kwargs)

    for i, (ims, lbls) in enumerate(loader):
        ims = ims.to(devices)
        lbls = lbls.to(devices)

        output = model(ims)
        probs = nn.Softmax(dim=1)(output)
        preds = torch.max(probs, 1)[1].detach().cpu().numpy()

        if loss_type == 'entropydice':
            pass
        elif loss_type == 'crossentropy':
            pass
        else:
            raise Exception (f'{loss_type} is not option')

        optimizer.zero_grad()
        loss = lossFun(output, lbls)
        loss.backward()
        optimizer.step()
        metrics.update(lbls.detach().cpu().numpy(), preds)
        running_loss += loss.item() * ims.size(0)

    scheduler.step()
    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss

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

    ##################################################
    ### (1) Load datasets                          ###
    ##################################################
    train_dst, val_dst, test_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size, 
                                num_workers=opts.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                num_workers=opts.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dst, batch_size=opts.test_batch_size, 
                                num_workers=opts.num_workers, shuffle=True, drop_last=True)

    ##################################################
    ### (2) Load models                            ###
    ##################################################
    segnet, vitnet = load_model(opts, verbose=True)

    ##################################################
    ### (3) Set up criterion                       ###
    ##################################################

    ''' depreciated
    '''
    ##################################################
    ### (4) Set up optimizer and scheduler         ###
    ##################################################
    Soptim, Ssche = set_optim(opts, opts.model, segnet)
    Voptim, Vsche = set_optim(opts, opts.vit_model, vitnet)

    ##################################################
    ### (5) Resume models, schedulers and optimizer ##
    ##################################################
    if opts.resume:
        raise NotImplementedError
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print('cuda multiple GPUs')
        segnet = nn.DataParallel(segnet)
        vitnet = nn.DataParallel(vitnet)

    segnet.to(devices)
    vitnet.to(devices)

    ##################################################
    ### (6) Set up metrics                         ###
    ##################################################

    ##################################################
    ### (7) Train                                  ###
    ##################################################
    for epoch in range(resume_epoch, opts.total_itrs):
        score, epoch_loss = train(devices=devices, model=segnet, loader=train_loader, 
                                loss_type=opts.loss_type, optimizer=Soptim, scheduler=Ssche, metrics= )
        
