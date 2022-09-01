import os
import socket
import argparse
from datetime import datetime

import network
import dataloader
import utils

HOSTNAME = {
    "server6" : 6,
    "server5" : 5
}
LOGIN = {
    "server6" : "/home/dongik/src/login.json",
    "server5" : "/home/dongik/src/login.json"
}
DEFAULT_PREFIX = {
    "server6" : "/DATA/dongik",
    "server5" : "/data1/sdi"
}
DATA_DIR = {
    "server6" : "/home/dongik/datasets",
    "server5" : "/home/dongik/datasets"
}


def _get_argparser():
    parser = argparse.ArgumentParser()

    # prefix & path options
    parser.add_argument("--short_memo", type=str, default='short memo',
                        help="breif explanation of experiment (default: short memo")
    parser.add_argument("--cur_work_server", type=int, default=0,
                        help="current working server (default: 0)")
    parser.add_argument("--default_prefix", type=str, default='/',
                        help="path to results directory (default: /)")
    parser.add_argument("--current_time", type=str, default='current_time',
                        help="results images folder name (default: current_time)")
    parser.add_argument("--data_root", type=str, default='/',
                        help="path to Dataset root directory (default: /)")
    parser.add_argument("--login_dir", type=str, default='/',
                        help="E-mail log-in info json file (default: /)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="gpus (default: 0)")

    # Tensorboard & store params options
    parser.add_argument("--Tlog_dir", type=str, default='/',
                        help="path to tensorboard log (default: /)")
    parser.add_argument("--save_model", action='store_false',
                        help="save best model param to \"./best-param\" (default: True)")
    parser.add_argument("--best_ckpt", type=str, default=None,
                        help="save best model param to \"./best-param\"")
    # Resume model from checkpoint
    parser.add_argument("--resume", action='store_true',
                        help="resume from checkpoint (defaults: false)")
    parser.add_argument("--resume_ckpt", default='/', type=str,
                        help="resume from checkpoint (defalut: /)")
    parser.add_argument("--continue_training", action='store_true',
                        help="restore state from reserved params (defaults: false)")

    # Model options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name]) )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50', choices=available_models,
                        help='semantic-segmentation model name (default: deeplabv3plus_resnet50)')
    parser.add_argument("--model_params", type=str, default='/',
                        help="pretrained semantic-segmentation model params (default: '/')")
    parser.add_argument("--model_pretrain", action='store_true',
                        help="restore param from checkpoint (defaults: false)")
    parser.add_argument("--vit_model", type=str, default='vit', choices=available_models,
                        help='vit model name (default: vit)')
    parser.add_argument("--vit_model_params", type=str, default='/',
                        help="pretrained vit model params (default: '/')")
    parser.add_argument("--vit_model_pretrain", action='store_true',
                        help="restore param from checkpoint (defaults: false)")
    # DeeplabV3+ options
    parser.add_argument("--encoder_name", type=str, default='resnet50',
                        help='Name of the classification model that will be used as an encoder (a.k.a backbone)')
    parser.add_argument("--encoder_depth", type=int, default=5,
                        help='A number of stages used in encoder in range [3, 5]')
    parser.add_argument("--encoder_weights", type=str, default='imagenet',
                        help=' One of None (random initialization), “imagenet” (pre-training on ImageNet) and other pretrained weights')
    parser.add_argument("--encoder_output_stride", type=int, default=16,
                        help='Downsampling factor for last encoder features')
    parser.add_argument("--decoder_atrous_rates", type=int, default=(12, 24, 36),
                        help='Dilation rates for ASPP module')
    parser.add_argument("--decoder_channels", type=tuple, default=256,
                        help='A number of convolution filters in ASPP module')
    parser.add_argument("--in_channels", type=int, default=3,
                        help='A number of input channels for the model, default is 3 (RGB images)')
    parser.add_argument("--classes", type=int, default=2,
                        help='A number of classes for output mask')
    parser.add_argument("--activation", type=str, default=None,
                        help='An activation function to apply after the final convolution layer')
    parser.add_argument("--upsampling", type=int, default=4,
                        help='Final upsampling factor. Default is 4 to preserve input-output spatial shape identity')
    parser.add_argument("--aux_params", type=dict, default=None,
                        help='Dictionary with parameters of the auxiliary output (classification head)')
    # ViT options
    parser.add_argument("--vit_image_size", type=int, default=512,
                        help='ViT image size')
    parser.add_argument("--vit_patch_size", type=int, default=64,
                        help='ViT patch size')
    parser.add_argument("--vit_num_classes", type=int, default=64,
                        help='ViT num classes')
    parser.add_argument("--vit_dim", type=int, default=1024,
                        help='ViT dim')
    parser.add_argument("--vit_depth", type=int, default=6,
                        help='ViT depth')
    parser.add_argument("--vit_heads", type=int, default=16,
                        help='ViT heads')
    parser.add_argument("--vit_mlp_dim", type=int, default=2048,
                        help='ViT mlp dim')
    parser.add_argument("--vit_dropout", type=float, default=0.1,
                        help='ViT dropout')
    parser.add_argument("--vit_emb_dropout", type=float, default=0.1,
                        help='ViT emb dropout')    

    # Dataset options
    available_datasets = sorted( name for name in dataloader.loader.__dict__ if  callable(dataloader.loader.__dict__[name]) )
    parser.add_argument("--dataset", type=str, default="cpn_vit", choices=available_datasets,
                        help='primary dataset (default: cpn_vit)')                  
    parser.add_argument("--dataset_ver", type=str, default="splits/v5/3",
                        help="version of primary dataset (default: splits/v5/3)")                   
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers (default: 8)")
    parser.add_argument("--tvs", type=int, default=20,
                        help="number of blocks in train set to be splited (default: 20)")

    # Transformation & Augmentation options
    parser.add_argument("--is_resize", action='store_true',
                        help="resize data (default: false)")
    parser.add_argument("--is_resize_val", action='store_true',
                        help="resize validate data (default: false)")
    parser.add_argument("--is_resize_test", action='store_true',
                        help="resize test data (default: false)")
    parser.add_argument("--resize", default=(256, 256))
    parser.add_argument("--resize_val", default=(256, 256))
    parser.add_argument("--resize_test", default=(256, 256))
    
    # Scale
    parser.add_argument("--is_scale", action='store_true',
                        help="scale data (default: false)")
    parser.add_argument("--is_scale_val", action='store_true',
                        help="scale data (default: false)")
    parser.add_argument("--is_scale_test", action='store_true',
                        help="scale data (default: false)")
    parser.add_argument("--scale_factor", type=float, default=5e-1)
    parser.add_argument("--scale_factor_val", type=float, default=5e-1)
    parser.add_argument("--scale_factor_test", type=float, default=5e-1)

    # Crop
    parser.add_argument("--is_crop", action='store_true',
                        help="crop data (default: false)")
    parser.add_argument("--is_crop_val", action='store_true',
                        help="crop data (default: false)")
    parser.add_argument("--is_crop_test", action='store_true',
                        help="crop data (default: false)")
    parser.add_argument("--crop_size", default=(256, 256))
    parser.add_argument("--crop_size_val", default=(256, 256))
    parser.add_argument("--crop_size_test", default=(256, 256))
    
    # Gaussian perturbation
    parser.add_argument("--std", type=float, default=0.0,
                        help="train sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="train mean in gaussian perturbation (default: 0)")
    parser.add_argument("--std_val", type=float, default=0.0,
                        help="val sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu_val", type=float, default=0.0,
                        help="val mean in gaussian perturbation (default: 0)")                
    parser.add_argument("--std_test", type=float, default=0.0,
                        help="test sigma in gaussian perturbation (default: 0)")
    parser.add_argument("--mu_test", type=float, default=0.0,
                        help="test mean in gaussian perturbation (default: 0)") 
    
    # Train options
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--total_itrs", type=int, default=2500,
                        help="epoch number (default: 2.5k)")
    parser.add_argument("--lr_policy", type=str, default='poly',
                        help="learning rate scheduler policy")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 100)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--optim", type=str, default='SGD',
                        help="optimizer (default: SGD)")
    parser.add_argument("--loss_type", type=str, default='entropydice',
                        help="criterion (default: ce+dl)")
    parser.add_argument("--vit_loss_type", type=str, default='crossentropy',
                        help="vit criterion (default: CrossEntropy)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("--vit_batch_size", type=int, default=32,
                        help='vit batch size (default: 32)')
    parser.add_argument("--exp_itr", type=int, default=20,
                        help='repeat N-identical experiments (default: 20)')

    # Validate options
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validate (default: 4)') 
    parser.add_argument("--val_vit_batch_size", type=int, default=4,
                        help='batch size for vit validate (default: 4)') 

    # Early stop options
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement after which training will be stopped (default: 100)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")
    
    # Test options
    parser.add_argument("--test_interval", type=int, default=1,
                        help="epoch interval for test (default: 1)")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help='batch size for test (default: 4)')
    parser.add_argument("--test_vit_batch_size", type=int, default=4,
                        help='vit batch size for test (default: 4)')
    parser.add_argument("--save_test_results", action='store_false',
                        help='save test results to \"./test\" (default: True)')
    parser.add_argument("--test_results_dir", type=str, default='/',
                        help="save segmentation results to (default: /)")
    
    # Run Demo
    parser.add_argument("--run_demo", action='store_true')

    return parser


def get_argparser(verbose=False):
    parser = _get_argparser().parse_args()

    hostname = socket.gethostname()
    s_folder = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + ('_demo' if parser.run_demo else '')
    _dir = os.path.join(DEFAULT_PREFIX[hostname], s_folder, current_time)

    if verbose:
        print(f'hostname: {hostname}\nfolder: {s_folder}\ncurrent time: {current_time}')

    if hostname not in HOSTNAME.keys():
        raise RuntimeError (f'hostname not found {hostname}')   
    
    parser.cur_work_server = HOSTNAME[hostname]
    parser.default_prefix = os.path.join(DEFAULT_PREFIX[hostname], s_folder)
    parser.current_time = current_time
    parser.data_root = DATA_DIR[hostname]
    parser.login_dir = LOGIN[hostname]
    parser.Tlog_dir = os.path.join(DEFAULT_PREFIX[hostname], s_folder, current_time, 'log')
    
    if not os.path.exists(parser.Tlog_dir):
        os.makedirs(parser.Tlog_dir)

    if parser.save_test_results:
        parser.test_results_dir = os.path.join(_dir, 'test')
        os.mkdir(parser.test_results_dir)

    if parser.save_model:
        parser.best_ckpt = os.path.join(_dir, 'best-param')
        os.mkdir(parser.best_ckpt)
    else:
        parser.best_ckpt = os.path.join(_dir, 'cache-param')
        os.mkdir(parser.best_ckpt)
        
    return parser

def save_argparser(parser, save_dir) -> dict:

    jsummary = {}
    for key, val in vars(parser).items():
        jsummary[key] = val

    utils.save_dict_to_json(jsummary, os.path.join(save_dir, 'summary.json'))

    return jsummary


if __name__ == "__main__":

    import utils

    print('basename:    ', os.path.basename(__file__)) # main.py
    print('dirname:     ', os.path.dirname(__file__)) # /data1/sdi/CPNKD
    print('abspath:     ', os.path.abspath(__file__)) # /data1/sdi/CPNKD/main.py
    print('abs dirname: ', os.path.dirname(os.path.abspath(__file__))) # /data1/sdi/CPNKD

    opts = get_argparser(verbose=True)
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val

    utils.save_dict_to_json(d=jsummary, json_path='/home/dongik/src/json-output/opts.json')

    pram = utils.Params('/home/dongik/src/json-output/opts.json')

    print(type(pram.decoder_channels)) # bool
    print(type(pram.classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.best_ckpt)) # str

    pram.update(json_path='/home/dongik/src/json-output/opts.json')
    utils.save_dict_to_json(pram.__dict__, '/home/dongik/src/json-output/out.json')