import os
import sys
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    from train_vcs import get_dataset, crop
    from args_vcs import get_argparser

    opts = get_argparser(verbose=True) 
    
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (devices, opts.gpus))
    opts.dataset = 'cpn'
    dst, _, _ = get_dataset(opts)

    dst_loader = DataLoader(dst, batch_size=64, 
                                num_workers=opts.num_workers, shuffle=True, drop_last=True)

    for i, (ims, lbls) in enumerate(dst_loader):
        ims = ims.to(devices)
        bboxs = lbls[0].to(devices)
        masks = lbls[1].to(devices)

        print(ims.shape)
        print(bboxs.shape)
        print(masks.shape)

        img, ma  = crop(ims.detach().cpu().numpy(), bboxs.detach().cpu().numpy(), masks.detach().cpu().numpy(), opts.vit_patch_size)


        break