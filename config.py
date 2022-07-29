import torch


class CFG:
    seed = 101
    debug = False
    exp_name = '2.5D'
    comment = 'unetplusplus - regnet_008'
    model_name = 'UNetPlusPlus'
    backbone = 'timm-regnety_008'
    train_bs = 32
    valid_bs = train_bs * 2
    img_size = [160, 192]
    epochs = 5
    lr = 2e-3
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 64 / train_bs)
    n_fold = 5
    folds = [0]
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
