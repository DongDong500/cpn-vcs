import datasets as dt
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

    print("Dataset: %s\n\tTrain\t%d\n\tVal\t%d\n\tTest\t%d" % 
            (dver + '/' + dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_dst, val_dst, test_dst


if __name__ == "__main__":
    '''
        test code
    '''