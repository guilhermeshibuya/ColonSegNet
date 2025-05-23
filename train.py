
import os
import time
import random
import numpy as np
from glob import glob
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import load_data, KvasirDataset
from utils import (
    seeding,shuffling, make_channel_first, make_channel_last, create_dir, epoch_time, print_and_save
)
from model import CompNet
from loss import DiceLoss, DiceBCELoss, IoUBCELoss
from metrics import DiceHelper, IouHelper, RecallHelper, PrecisionHelper

def train(model, loader, optimizer, loss_fn, device):
    dice_helper = DiceHelper()
    iou_helper = IouHelper()
    recall_helper = RecallHelper()
    precision_helper = PrecisionHelper()

    epoch_loss = 0
    model.train()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        yp = model(x)
        loss = loss_fn(yp, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        yp_sigmoid = torch.sigmoid(yp)
        yp_bin = (yp_sigmoid > 0.5).cpu().numpy()
        y_np = (y.cpu().numpy() > 0.5).astype(np.uint8)

        for j in range(x.size(0)):
            iou_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
            dice_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
            recall_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
            precision_helper.add_masks(yp_bin[j, 0], y_np[j, 0])

    epoch_loss = epoch_loss/len(loader)

    dice_score = dice_helper.calculate_dice()
    iou_score = iou_helper.calculate_iou()
    recall_score = recall_helper.calculate_recall()
    precision_score = precision_helper.calculate_precision()

    return epoch_loss, dice_score, iou_score, recall_score, precision_score

def evaluate(model, loader, loss_fn, device):
    dice_helper = DiceHelper()
    iou_helper = IouHelper()
    recall_helper = RecallHelper()
    precision_helper = PrecisionHelper()

    epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            yp = model(x)
            loss = loss_fn(yp, y)
            epoch_loss += loss.item()

            yp_sigmoid = torch.sigmoid(yp)
            yp_bin = (yp_sigmoid > 0.5).cpu().numpy()
            y_np = (y.cpu().numpy() > 0.5).astype(np.uint8)

            for j in range(x.size(0)):
                iou_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
                dice_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
                recall_helper.add_masks(yp_bin[j, 0], y_np[j, 0])
                precision_helper.add_masks(yp_bin[j, 0], y_np[j, 0])

    epoch_loss = epoch_loss/len(loader)

    dice_score = dice_helper.calculate_dice()
    iou_score = iou_helper.calculate_iou()
    recall_score = recall_helper.calculate_recall()
    precision_score = precision_helper.calculate_precision()

    return epoch_loss, dice_score, iou_score, recall_score, precision_score

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    files_path = ""
    create_dir(files_path)

    """ Training logfile """
    train_log_path = os.path.join(files_path, "train_log.txt")
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    """ Load dataset """
    dataset_path = ""
    (train_x, train_y), (valid_x, valid_y), _ = load_data(dataset_path)

    # Augmented dataset
    # augmented_path ="new_data"
    # (train_x, train_y), (valid_x, valid_y), _ = load_data(augmented_path)

    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 4
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "./checkpoint.pth"
    best_model_path = os.path.join(files_path, "best.pth") 
    last_model_path = os.path.join(files_path, "last.pth")

    """ Dataset and loader """
    train_dataset = KvasirDataset(train_x, train_y, size)
    valid_dataset = KvasirDataset(valid_x, valid_y, size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = CompNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    # loss_fn = IoUBCELoss()
    loss_name = "BCE Dice Loss"

    start_epoch = 0
    best_valid_loss = float('inf')

    if os.path.exists(last_model_path):
        print('Loading checkpoint...')
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint['best_valid_loss']
        print(f"Checkpoint loaded. Starting from epoch {start_epoch}")


    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_dice, train_iou, train_recall, train_precision = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, val_dice, val_iou, val_recall, val_precision = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_valid_loss': best_valid_loss,
        }, last_model_path)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
            }, best_model_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f} | Train Dice: {train_dice:.3f} | Train IoU: {train_iou:.3f} | Train Recall: {train_recall:.3f} | Train Precision: {train_precision:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f} |  Val. Dice: {val_dice:.3f} |  Val. IoU: {val_iou:.3f} |  Val. Recall: {val_recall:.3f} |  Val. Precision: {val_precision:.3f}\n'
        print_and_save(train_log_path, data_str)
