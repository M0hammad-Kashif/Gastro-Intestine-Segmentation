import copy
import gc
import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim import optimizer

from config import CFG
from dataloader import valid_loader, train_loader
from imports import amp, tqdm, c_, sr_
from loss import criterion, dice_coef, iou_coef
from utils import fold


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss


def valid_one_epoch(model, dataloader, device, epoch):
    with torch.inference_mode():
        model.eval()

        dataset_size = 0
        running_loss = 0.0

        val_scores = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            y_pred = model(images)
            loss = criterion(y_pred, masks)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            y_pred = nn.Sigmoid()(y_pred)
            val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
            val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                             lr=f'{current_lr:0.5f}',
                             gpu_memory=f'{mem:0.2f} GB')
        val_scores = np.mean(val_scores, axis=0)
        torch.cuda.empty_cache()
        gc.collect()

        return epoch_loss, val_scores


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=CFG.device, epoch=epoch)

        val_loss, val_scores = valid_one_epoch(model, valid_loader,
                                               device=CFG.device,
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        # Log the metrics
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            wandb.run.summary["Best Dice"] = best_dice
            wandb.run.summary["Best Jaccard"] = best_jaccard
            wandb.run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)

        print();
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
