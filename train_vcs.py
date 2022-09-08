import os
import random
import numbers
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
import network
import criterion
import dataloader
from metrics import StreamSegMetrics
from metrics import ClsMetrics
from utils import ext_transforms as et

def get_class_weight(lbls):
    '''
    compute class weight (only binary)
        Args:
            lbls (numpy array)
        Returns:
            weight (numpy array)
    '''
    weights = lbls.sum() / (lbls.shape[0] * lbls.shape[1] * lbls.shape[2])
    
    if weights < 0 or weights > 1:
        raise Exception (f'weights: {weights} for cross entropy is wrong')

    return [weights, 1 - weights]

def add_writer_scalar(writer, phase, score, loss, epoch):
    writer.add_scalar(f'IoU BG/{phase}', score['Class IoU'][0], epoch)
    writer.add_scalar(f'IoU Nerve/{phase}', score['Class IoU'][1], epoch)
    writer.add_scalar(f'Dice BG/{phase}', score['Class F1'][0], epoch)
    writer.add_scalar(f'Dice Nerve/{phase}', score['Class F1'][1], epoch)
    writer.add_scalar(f'epoch loss/{phase}', loss, epoch)

def add_writer_vit_scalar(writer, phase, score, loss, epoch, total_itrs, num_classes):
    
    writer.add_scalar(f'vit/epoch loss/cls {num_classes}/{phase}', loss, epoch)
    writer.add_scalar(f'vit/epoch Overall Acc/cls {num_classes}/{phase}', score['Overall Acc'], epoch)
    
    for i in range(num_classes):
        writer.add_scalar(f'vit/f1 score/cls {num_classes}/{i}/{phase}', score['Class F1'][i], epoch)

    print(f'[{phase}-vit]\t {epoch}/{total_itrs} Overall Acc: {score["Overall Acc"]:.4f}, loss: {loss:.4f}')

def print_result(phase, score, epoch, total_itrs, loss):
    print("[{}] Epoch: {}/{} Loss: {:.5f}".format(phase, epoch, total_itrs, loss))
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))

def get_dataset(opts, run_id):
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
        cur_time=opts.current_time, rid=run_id,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='train', transform=train_transform, in_channels=opts.in_channels,
        image_patch_size=(opts.vit_patch_size, opts.vit_patch_size))

    val_dst = dataloader.loader.__dict__[opts.dataset]( 
        pth = os.path.join(opts.data_root, opts.dataset, opts.dataset_ver), tvs = opts.tvs, mkset = False,
        cur_time = opts.current_time, rid=run_id,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='val', transform=val_transform, in_channels=opts.in_channels,
        image_patch_size=(opts.vit_patch_size, opts.vit_patch_size) )

    test_dst = dataloader.loader.__dict__[opts.dataset]( 
        pth = os.path.join(opts.data_root, opts.dataset, opts.dataset_ver), tvs = opts.tvs, mkset = False,
        cur_time = opts.current_time, rid=run_id,
        root=opts.data_root, datatype=opts.dataset, dver=opts.dataset_ver, 
        image_set='test', transform=test_transform, in_channels=opts.in_channels,
        image_patch_size=(opts.vit_patch_size, opts.vit_patch_size) )

    print("Dataset - %s\n\tTrain\t%d\n\tVal\t%d\n\tTest\t%d" % 
            (opts.dataset_ver + '/' + opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_dst, val_dst, test_dst

def load_model(opts, verbose = True):

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
        optimizer = torch.optim.SGD(
            model.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    return optimizer, scheduler

def crop(ims, bboxs, mas, patch_size, crop_size=256):
    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size
    
    cims = np.zeros((ims.shape[0], 3, crop_size[0], crop_size[1]), dtype=ims.dtype)
    cmas = np.zeros((mas.shape[0], crop_size[0], crop_size[1]), dtype=mas.dtype)
    
    block = 512 / patch_size
    q = bboxs // block
    r = bboxs % block
    pnt = (( (q * patch_size + (q+1) * patch_size)/2 ).astype(int), ( (r * patch_size + (r+1) * patch_size)/2 ).astype(int))

    for i in range(ims.shape[0]):
        # Height
        if pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = [int(pnt[0][i] - crop_size[0]/2), 0]
            rb = [int(pnt[0][i] + crop_size[0]/2), 0]
        elif pnt[0][i] < crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = [0, 0]
            rb = [crop_size[0], 0]
        elif pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) < crop_size[0]/2:
            lt = [512 - crop_size[0], 0]
            rb = [512, 0]
        # Width
        if pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = int(pnt[1][i] - crop_size[1]/2)
            rb[1] = int(pnt[1][i] + crop_size[1]/2)
        elif pnt[1][i] < crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = 0
            rb[1] = crop_size[1]
        elif pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) < crop_size[1]/2:
            lt[1] = 512 - crop_size[1]
            rb[1] = 512
        
        cims[i, ...] = ims[i, ...][... , lt[0]:rb[0], lt[1]:rb[1]]
        cmas[i, ...] = mas[i, ...][lt[0]:rb[0], lt[1]:rb[1]]

        from PIL import Image
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        def cmap(N=3, preds=False):
            color_map = np.zeros((N, 3), dtype='uint8')
            color_map[0] = np.array([255, 255, 255])
            color_map[1] = np.array([255, 0, 0]) if preds else np.array([0, 0, 255])
            color_map[2] = np.array([50, 200, 100])
            return color_map

        #cmp = cmap(N=3, preds=False)
        #tar = (denorm(cims[i, ...]) * 255).transpose(1, 2, 0) * 0.5 + cmp[cmas[i, ...]] * 0.5
        #Image.fromarray((tar).astype(np.uint8), 'RGB').save('/home/dongik/src/final.bmp')
    
    return torch.from_numpy(cims), torch.from_numpy(cmas).type(torch.long)

def _recover(ims, bboxs, masks, patch_size, crop_size=256):
    """
    Args:
        ims (numpy.ndarray) size : (B x C x H x W)
        lbls (numpy.ndarray) lbls[0] : size (B), lbls[1] : size (B x H x W)
        cls (numpy.ndarray) size : (B)
    """
    from PIL import Image
    def cmap(N=3, preds=False):
        color_map = np.zeros((N, 3), dtype='uint8')
        color_map[0] = np.array([255, 255, 255])
        color_map[1] = np.array([255, 0, 0]) if preds else np.array([0, 0, 255])
        color_map[2] = np.array([50, 200, 100])
        return color_map
    cmp = cmap(N=3, preds=False)

    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size
    
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    block = 512 / patch_size
    
    q = bboxs // block
    r = bboxs % block
    pnt = (( (q * patch_size + (q+1) * patch_size)/2 ).astype(int), ( (r * patch_size + (r+1) * patch_size)/2 ).astype(int))
    print(pnt)
    #masks = masks.astype(np.uint8) * 255

    cimage = np.zeros((ims.shape[0], 3, crop_size[0], crop_size[1]), dtype='uint8')
    cmask = np.zeros((masks.shape[0], crop_size[0], crop_size[1]), dtype='uint8')
    for i in range(masks.shape[0]):
        tar1 = (denorm(ims[i]) * 255).transpose(1, 2, 0).astype(np.uint8)
        print(masks[i, ...])
        ma = cmp[masks[i, ...]]

        # Height
        if pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = (int(pnt[0][i] - crop_size[0]/2), 0)
            rb = (int(pnt[0][i] + crop_size[0]/2), 0)
        elif pnt[0][i] < crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = (0, 0)
            rb = (crop_size[0], 0)
        elif pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) < crop_size[0]/2:
            lt = (512 - crop_size[0], 0)
            rb = (512, 0)
        lt = list(lt)
        rb = list(rb)
        # Width
        if pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = int(pnt[1][i] - crop_size[1]/2)
            rb[1] = int(pnt[1][i] + crop_size[1]/2)
        elif pnt[1][i] < crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = 0
            rb[1] = crop_size[1]
        elif pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) < crop_size[1]/2:
            lt[1] = 512 - crop_size[1]
            rb[1] = 512

        Image.fromarray(tar1, 'RGB').save('/home/dongik/src/out.png')
        masks[i, ...][pnt[0][i]-3:pnt[0][i]+3, pnt[1][i]-int(patch_size/2):pnt[1][i]+int(patch_size/2)] = 2
        masks[i, ...][pnt[0][i]-int(patch_size/2):pnt[0][i]+int(patch_size/2), pnt[1][i]-3:pnt[1][i]+3] = 2
        
        masks[i, ...][lt[0]:rb[0], lt[1]:lt[1]+3] = 2
        masks[i, ...][lt[0]:rb[0], rb[1]-3:rb[1]] = 2
        masks[i, ...][lt[0]:lt[0]+3, lt[1]:rb[1]] = 2
        masks[i, ...][rb[0]-3:rb[0], lt[1]:rb[1]] = 2
        
        #cimage[i, ...] = ims[i, ...][:, lt[0]:rb[0], lt[1]:rb[1]]
        cimage[i, ...] = tar1[lt[0]:rb[0], lt[1]:rb[1], ...].transpose(2, 0, 1)
        cmask[i, ...] = masks[i, ...][lt[0]:rb[0], lt[1]:rb[1]]
        tar2 = (cimage[i]).transpose(1, 2, 0).astype(np.uint8)
        
        cSpot = (tar2 * 0.5 + cmp[cmask[i, ...]] * 0.5).astype(np.uint8)
        Image.fromarray(cSpot, 'RGB').save('/home/dongik/src/cSpot.bmp')
        Image.fromarray(cSpot, 'RGB').save('/home/dongik/src/cSpot.png')

        maSpot = (tar1 * 0.5 + cmp[masks[i, ...]] * 0.5).astype(np.uint8)

        print(maSpot.shape)
        
        Image.fromarray(maSpot, 'RGB').save('/home/dongik/src/maSpot.bmp')
        Image.fromarray(maSpot, 'RGB').save('/home/dongik/src/maSpot.png')

    return ims, bboxs, masks

def recover(mask_shape, anchor, cmask, patch_size, crop_size=256, print_anchor=False):
    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size

    block = 512 / patch_size
    q = anchor // block
    r = anchor % block
    pnt = (( (q * patch_size + (q+1) * patch_size)/2 ).astype(int), ( (r * patch_size + (r+1) * patch_size)/2 ).astype(int))

    cmask = cmask.astype(np.uint8)
    overlap = np.zeros(mask_shape, dtype='uint8')
    for i in range(mask_shape[0]):
        # Height
        if pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = [int(pnt[0][i] - crop_size[0]/2), 0]
            rb = [int(pnt[0][i] + crop_size[0]/2), 0]
        elif pnt[0][i] < crop_size[0]/2 and (512 - pnt[0][i]) >= crop_size[0]/2:
            lt = [0, 0]
            rb = [crop_size[0], 0]
        elif pnt[0][i] >= crop_size[0]/2 and (512 - pnt[0][i]) < crop_size[0]/2:
            lt = [512 - crop_size[0], 0]
            rb = [512, 0]
        # Width
        if pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = int(pnt[1][i] - crop_size[1]/2)
            rb[1] = int(pnt[1][i] + crop_size[1]/2)
        elif pnt[1][i] < crop_size[1]/2 and (512 - pnt[1][i]) >= crop_size[1]/2:
            lt[1] = 0
            rb[1] = crop_size[1]
        elif pnt[1][i] >= crop_size[1]/2 and (512 - pnt[1][i]) < crop_size[1]/2:
            lt[1] = 512 - crop_size[1]
            rb[1] = 512

        if mask_shape == cmask.shape:
            overlap[i, ...] += cmask[i, ...]
        else:
            overlap[i, ...][lt[0]:rb[0], lt[1]:rb[1]] += cmask[i, ...]

        if print_anchor:
            overlap[i, ...][pnt[0][i]-3:pnt[0][i]+3, pnt[1][i]-int(patch_size/2):pnt[1][i]+int(patch_size/2)] = 2
            overlap[i, ...][pnt[0][i]-int(patch_size/2):pnt[0][i]+int(patch_size/2), pnt[1][i]-3:pnt[1][i]+3] = 2
            overlap[i, ...][lt[0]:rb[0], lt[1]:lt[1]+3] = 2
            overlap[i, ...][lt[0]:rb[0], rb[1]-3:rb[1]] = 2
            overlap[i, ...][lt[0]:lt[0]+3, lt[1]:rb[1]] = 2
            overlap[i, ...][rb[0]-3:rb[0], lt[1]:rb[1]] = 2

    return overlap

def train(devices, loader, Snet, Vnet, nSloss, nVloss,
            Soptimizer, Sscheduler, Voptimizer, Vscheduler, 
            patch_size, crop_size, seg_metrics, vit_metrics, 
            use_true_anchor, ewu, **kwargs):

    Snet.train()
    Vnet.train()

    vit_metrics.reset()
    seg_metrics.reset()
    vit_running_loss = 0.0
    seg_running_loss = 0.0

    Sloss = criterion.get_criterion.__dict__[nSloss](**kwargs)
    Vloss = criterion.get_criterion.__dict__[nVloss](**kwargs)

    for i, (ims, lbls) in enumerate(loader):
        image = ims.to(devices)
        anchor = lbls[0].type(torch.long).to(devices)
        mask = lbls[1].cpu().numpy()

        ### ViT
        Voutput = Vnet(image)
        Vprob = nn.Softmax(dim=1)(Voutput)
        anchor_pred = torch.max(Vprob, 1)[1].detach().cpu().numpy()
        anchor_true = anchor.detach().cpu().numpy()
        vit_metrics.update(anchor_true, anchor_pred)

        Voptimizer.zero_grad()
        if nVloss == 'crossentropy':
            vit_loss = Vloss(Voutput, anchor)
        else:
            raise Exception (f'{nVloss} is not option')
        vit_loss.backward()
        Voptimizer.step()
        vit_running_loss += vit_loss.item() * image.size(0)

        if use_true_anchor:
            cimage, cmask = crop(image.detach().cpu().numpy(), anchor_true, 
                                mask, patch_size, crop_size)
        else:
            cimage, cmask = crop(image.detach().cpu().numpy(), anchor_pred, 
                                mask, patch_size, crop_size)
        cimage = cimage.to(devices)
        cmask = cmask.to(devices)

        ### Segmentation
        Soutput = Snet(cimage)
        Sprob = nn.Softmax(dim=1)(Soutput)
        seg_pred = torch.max(Sprob, 1)[1].detach().cpu().numpy()
        seg_true = cmask.detach().cpu().numpy()

        if use_true_anchor:
            overlay = recover(mask.shape, anchor_true, seg_pred, patch_size, crop_size, False)
        else:
            overlay = recover(mask.shape, anchor_pred, seg_pred, patch_size, crop_size, False)
        seg_metrics.update(mask, overlay)

        Soptimizer.zero_grad()
        if nSloss == 'entropydice' and ewu:
            cls_weight = torch.tensor(get_class_weight(seg_true), dtype=torch.float32).to(devices)
            Sloss.update_weight(weight=cls_weight)
            seg_loss = Sloss(Soutput, cmask)
        elif nSloss == 'entropydice':
            seg_loss = Sloss(Soutput, cmask)
        else:
            raise Exception (f'{nSloss} is not option')
        seg_loss.backward()
        Soptimizer.step()
        seg_running_loss += seg_loss.item() * cimage.size(0)

    Sscheduler.step()
    Vscheduler.step()

    seg_epoch_loss = seg_running_loss / len(loader.dataset)
    vit_epoch_loss = vit_running_loss / len(loader.dataset)

    seg_score = seg_metrics.get_results()
    vit_score = vit_metrics.get_results()

    return seg_score, vit_score, seg_epoch_loss, vit_epoch_loss

def validate(devices, loader, Snet, Vnet, nSloss, nVloss,
            patch_size, crop_size, seg_metrics, vit_metrics,
            use_true_anchor, ewu, **kwargs):

    Snet.eval()
    Vnet.eval()

    vit_metrics.reset()
    seg_metrics.reset()
    vit_running_loss = 0.0
    seg_running_loss = 0.0

    Sloss = criterion.get_criterion.__dict__[nSloss](**kwargs)
    Vloss = criterion.get_criterion.__dict__[nVloss](**kwargs)

    with torch.no_grad():
        for i, (ims, lbls) in enumerate(loader):
            image = ims.to(devices)
            anchor = lbls[0].type(torch.long).to(devices)
            mask = lbls[1].cpu().numpy()

            ### ViT
            Voutput = Vnet(image)
            Vprob = nn.Softmax(dim=1)(Voutput)
            anchor_pred = torch.max(Vprob, 1)[1].detach().cpu().numpy()
            anchor_true = anchor.detach().cpu().numpy()
            vit_metrics.update(anchor_true, anchor_pred)

            if nVloss == 'crossentropy':
                vit_loss = Vloss(Voutput, anchor)
            else:
                raise Exception (f'{nVloss} is not option')
            vit_running_loss += vit_loss.item() * image.size(0)

            if use_true_anchor:
                cimage, cmask = crop(image.detach().cpu().numpy(), anchor_true, 
                                    mask, patch_size, crop_size)
            else:
                cimage, cmask = crop(image.detach().cpu().numpy(), anchor_pred, 
                                    mask, patch_size, crop_size)
            cimage = cimage.to(devices)
            cmask = cmask.to(devices)

            ### Segmentation
            Soutput = Snet(cimage)
            Sprob = nn.Softmax(dim=1)(Soutput)
            seg_pred = torch.max(Sprob, 1)[1].detach().cpu().numpy()
            seg_true = cmask.detach().cpu().numpy()

            if use_true_anchor:
                overlay = recover(mask.shape, anchor_true, seg_pred, patch_size, crop_size, False)
            else:
                overlay = recover(mask.shape, anchor_pred, seg_pred, patch_size, crop_size, False)
            seg_metrics.update(mask, overlay)

            if nSloss == 'entropydice' and ewu:
                cls_weight = torch.tensor(get_class_weight(seg_true), dtype=torch.float32).to(devices)
                Sloss.update_weight(weight=cls_weight)
                seg_loss = Sloss(Soutput, cmask)
            elif nSloss == 'entropydice':
                seg_loss = Sloss(Soutput, cmask)
            else:
                raise Exception (f'{nSloss} is not option')
            seg_running_loss += seg_loss.item() * cimage.size(0)

    seg_epoch_loss = seg_running_loss / len(loader.dataset)
    vit_epoch_loss = vit_running_loss / len(loader.dataset)

    seg_score = seg_metrics.get_results()
    vit_score = vit_metrics.get_results()

    return seg_score, vit_score, seg_epoch_loss, vit_epoch_loss

def save(devices, loader, Snet, Vnet,
            patch_size, crop_size, path, writer, run_id, use_true_anchor):
    from PIL import Image
    def cmap(N=3, pred=True):
        """
            Blue : Pred
            Red  : True
        """
        color_map = np.zeros((N, 3), dtype=np.uint8)
        color_map[0] = np.array([0, 0, 0])
        color_map[1] = np.array([0, 0, 255]) if pred else np.array([255, 0, 0])
        color_map[2] = np.array([153, 153, 255]) if pred else np.array([255, 153, 153])
        return color_map
    cmp_pred = cmap(N=3, pred=True)
    cmp_true = cmap(N=3, pred=False)

    try:
        utils.mkdir(path) if not os.path.exists(path) else 0
    except:
        raise Exception("Cannot make directory: {}".format(path))
    
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    Snet.eval()
    Vnet.eval()
    with torch.no_grad():
        for i, (ims, lbls) in enumerate(loader):
            image = ims.to(devices)
            anchor = lbls[0].type(torch.long).to(devices)
            mask = lbls[1].to(devices)

            ### ViT
            Voutput = Vnet(image)
            Vprob = nn.Softmax(dim=1)(Voutput)
            anchor_pred = torch.max(Vprob, 1)[1].detach().cpu().numpy()
            anchor_true = anchor.detach().cpu().numpy()

            if use_true_anchor:
                cimage, cmask = crop(image.detach().cpu().numpy(), anchor_true, 
                                    mask.detach().cpu().numpy(), patch_size, crop_size)
            else:
                cimage, cmask = crop(image.detach().cpu().numpy(), anchor_pred, 
                                    mask.detach().cpu().numpy(), patch_size, crop_size)
            cimage = cimage.to(devices)
            cmask = cmask.to(devices)

            ### Segmentation
            Soutput = Snet(cimage)
            Sprob = nn.Softmax(dim=1)(Soutput)
            seg_pred = torch.max(Sprob, 1)[1].detach().cpu().numpy()

            image = image.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            
            if use_true_anchor:
                pred_overlay = recover(mask.shape, anchor_true, seg_pred, patch_size, crop_size, True)
                true_overlay = recover(mask.shape, anchor_true, mask, patch_size, crop_size, True)
            else:
                pred_overlay = recover(mask.shape, anchor_pred, seg_pred, patch_size, crop_size, True)
                true_overlay = recover(mask.shape, anchor_true, mask, patch_size, crop_size, True)
            tar_img = (denorm(image) * 255).transpose(0, 2, 3, 1)
            tar_mask = (cmp_pred[pred_overlay] + cmp_true[true_overlay])
            result = (tar_img * 0.5 + tar_mask * 0.5).astype(np.uint8)
            writer.add_image(f'result_image/{run_id}', result,  global_step=i, walltime=0, dataformats='NHWC')

            for j in range(image.shape[0]):
                tar_img = (denorm(image[j]) * 255).transpose(1, 2, 0)
                tar_mask = (cmp_pred[pred_overlay[j]] + cmp_true[true_overlay[j]])
                result = (tar_img * 0.5 + tar_mask * 0.5).astype(np.uint8)
                idx = str(i * image.shape[0] + j).zfill(3)
                Image.fromarray(result, "RGB").save(os.path.join( path, '{}_overlay.png'.format(idx) ))
                


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
    train_dst, val_dst, test_dst = get_dataset(opts, RUN_ID)
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
    seg_metrics = StreamSegMetrics(n_classes=opts.classes)
    vit_metrics = ClsMetrics(n_classes=opts.vit_num_classes)
    seg_early_stopping = utils.DiceStopping(patience=opts.patience, delta=opts.delta, verbose=True,
                                        path=os.path.join(opts.best_ckpt, RUN_ID))
    vit_early_stopping = utils.EarlyStopping(patience=opts.patience, delta=opts.delta, verbose=True,
                                        path=os.path.join(opts.best_ckpt, RUN_ID))
    seg_best_epoch = 0
    seg_best_score = 0

    ##################################################
    ### (7) Train                                  ###
    ##################################################
    for epoch in range(resume_epoch, opts.total_itrs):
        seg_score, vit_score, seg_epoch_loss, vit_epoch_loss = train(
            devices=devices, loader=train_loader, 
            Snet=segnet, Vnet=vitnet, 
            nSloss=opts.loss_type, nVloss=opts.vit_loss_type, 
            Soptimizer=Soptim, Sscheduler=Ssche, 
            Voptimizer=Voptim, Vscheduler=Vsche,
            patch_size=opts.vit_patch_size, crop_size=opts.crop_size, 
            seg_metrics=seg_metrics, vit_metrics=vit_metrics,
            use_true_anchor=opts.use_true_anchor, ewu=opts.ewu,
            opts=opts 
            )
        print_result('train', seg_score, epoch, opts.total_itrs, seg_epoch_loss)
        add_writer_scalar(writer, 'train', seg_score, seg_epoch_loss, epoch)
        add_writer_vit_scalar(writer, 'train', vit_score, vit_epoch_loss, epoch, opts.total_itrs, opts.vit_num_classes) 

        val_seg_score, val_vit_score, val_seg_epoch_loss, val_vit_epoch_loss = validate(
            devices=devices, loader=val_loader, 
            Snet=segnet, Vnet=vitnet, 
            nSloss=opts.loss_type, nVloss=opts.vit_loss_type, 
            patch_size=opts.vit_patch_size, crop_size=opts.crop_size, 
            seg_metrics=seg_metrics, vit_metrics=vit_metrics,
            use_true_anchor=opts.use_true_anchor_val, ewu=opts.ewu,
            opts=opts 
            )
        print_result('val', val_seg_score, epoch, opts.total_itrs, val_seg_epoch_loss)
        add_writer_scalar(writer, 'val', val_seg_score, val_seg_epoch_loss, epoch)
        add_writer_vit_scalar(writer, 'val', val_vit_score, val_vit_epoch_loss, epoch, opts.total_itrs, opts.vit_num_classes)

        test_seg_score, test_vit_score, test_seg_epoch_loss, test_vit_epoch_loss = validate(
            devices=devices, loader=test_loader, 
            Snet=segnet, Vnet=vitnet, 
            nSloss=opts.loss_type, nVloss=opts.vit_loss_type, 
            patch_size=opts.vit_patch_size, crop_size=opts.crop_size, 
            seg_metrics=seg_metrics, vit_metrics=vit_metrics, 
            use_true_anchor=opts.use_true_anchor_val, ewu=opts.ewu,
            opts=opts 
            )
        print_result('test', test_seg_score, epoch, opts.total_itrs, test_seg_epoch_loss)
        add_writer_scalar(writer, 'test', test_seg_score, test_seg_epoch_loss, epoch)
        add_writer_vit_scalar(writer, 'test', test_vit_score, test_vit_epoch_loss, epoch, opts.total_itrs, opts.vit_num_classes)
        
        if seg_early_stopping(test_seg_score['Class F1'][1], segnet, Soptim, Ssche, epoch):
            seg_best_epoch = epoch
            seg_best_score = test_seg_score
        
        vit_early_stopping(test_vit_epoch_loss, vitnet, Voptim, Vsche, epoch)

        if seg_early_stopping.early_stop:
            print("Early Stop !!!")
            break

        if opts.run_demo and epoch > 0:
            print("Run demo !!!")
            break

    ##################################################
    ### (8) Save results                           ###
    ##################################################
    params = utils.Params(json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json')).dict
    params[f'{RUN_ID} dice score'] = seg_best_score["Class F1"][1]
    utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))

    checkpoint = torch.load(os.path.join(opts.best_ckpt, RUN_ID, 'dicecheckpoint.pt'), map_location=devices)
    segnet.load_state_dict(checkpoint["model_state"])
    checkpoint = torch.load(os.path.join(opts.best_ckpt, RUN_ID, 'checkpoint.pt'), map_location=devices)
    vitnet.load_state_dict(checkpoint["model_state"])

    sdir = os.path.join(opts.test_results_dir, RUN_ID, f'epoch_{seg_best_epoch}')
    save(devices, test_loader, segnet, vitnet, 
            patch_size=opts.vit_patch_size, crop_size=opts.crop_size, 
            path=sdir, writer=writer, run_id=RUN_ID, use_true_anchor=opts.use_true_anchor_val)

    del checkpoint
    del segnet
    del vitnet
    torch.cuda.empty_cache()

    return {
                'Model' : opts.model, 'Dataset' : opts.dataset,
                'OS' : str(opts.encoder_output_stride), 'Epoch' : str(seg_best_epoch),
                'F1 [0]' : seg_best_score['Class F1'][0], 'F1 [1]' : seg_best_score['Class F1'][1]
            }
