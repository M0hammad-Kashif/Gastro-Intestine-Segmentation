import wandb

from imports import *
from model import build_model
from optimizer import fetch_scheduler
from train_val import run_training
from utils import *

set_seed(CFG.seed)

model = build_model()
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)

for fold in CFG.folds:
    print(f'#' * 15)
    print(f'### Fold: {fold}')
    print(f'#' * 15)
    run = wandb.init(project='uw-maddison-gi-tract')
    train_loader, valid_loader = prepare_loaders(fold=fold, debug=CFG.debug)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CFG.device,
                                  num_epochs=CFG.epochs)
    run.finish()
    display(ipd.IFrame(run.url, width=1000, height=720))
