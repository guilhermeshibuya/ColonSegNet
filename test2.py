
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np

import cv2
import torch
from sklearn.metrics import (
    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)

from model import CompNet
from utils import create_dir, seeding, make_channel_last
from data import load_data
from crf import apply_crf
from metrics import IouHelper, DiceHelper, RecallHelper, PrecisionHelper

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jaccard_score(y_true, y_pred, average='binary')
    score_f1 = f1_score(y_true, y_pred, average='binary')
    score_recall = recall_score(y_true, y_pred, average='binary')
    score_precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    score_acc = accuracy_score(y_true, y_pred)
    score_fbeta = fbeta_score(y_true, y_pred, beta=1.0, average='binary', zero_division=1)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    path = "/content/drive/Shareddrives/Projeto Zscan 2 - datasets segmentacao/datasets/segmentação/Polypgen/sequence/positive/splitted"
    (train_x, train_y), _, (test_x, test_y) = load_data(path)

    """ Hyperparameters """
    size = (512, 512)
    checkpoint_path = "files/checkpoint.pth"

    """ Directories """
    create_dir("results/mix")
    create_dir("results/mask")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Testing """
    time_taken = []

    iou_helper = IouHelper()
    dice_helper = DiceHelper()
    recall_helper = RecallHelper()
    precision_helper = PrecisionHelper()

    for i, (x, y) in enumerate(zip(test_x, test_y)):
        image_name = y.split("/")[-1]
        mask1 = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.resize(mask1, size)
        mask1 = mask1 / 255.0
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = torch.tensor(mask1, dtype=torch.float32).to(device)

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, size)
        x = x / 255.0
        x = x.transpose(2, 0, 1)
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float32).to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y1 = model(x)
            pred_y1 = torch.sigmoid(pred_y1)
            pred_y1 = pred_y1[0].cpu().numpy()
            pred_y1 = np.squeeze(pred_y1)
            stop_time = time.time()

            total_time = stop_time - start_time
            time_taken.append(total_time)

            pred_y1_bin = pred_y1 > 0.5
            pred_y1_bin = pred_y1_bin.astype(np.uint8)
            mask1_bin = mask1[0][0].cpu().numpy().astype(np.uint8)

            iou_helper.add_masks(mask1_bin, pred_y1_bin)
            dice_helper.add_masks(mask1_bin, pred_y1_bin)
            recall_helper.add_masks(mask1_bin, pred_y1_bin)
            precision_helper.add_masks(mask1_bin, pred_y1_bin)

        ori_img1 = ori_img1
        ori_mask1 = mask_parse(ori_mask1)
        pred_y1 = mask_parse(pred_y1)
        sep_line = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img1, sep_line,
            ori_mask1, sep_line,
            pred_y1
        ]

        cat_images = np.concatenate(tmp, axis=1)
        cv2.imwrite(f"results/mix/{image_name}.png", cat_images)
        cv2.imwrite(f"results/mask/{image_name}.png", pred_y1)

    mean_iou = iou_helper.calculate_iou()
    mean_dice = dice_helper.calculate_dice()
    mean_recall = recall_helper.calculate_recall()
    mean_precision = precision_helper.calculate_precision()
    avg_time = np.mean(time_taken)
    fps = 1 / avg_time if avg_time > 0 else 0

    print(f"IoU: {mean_iou:.4f} - Dice: {mean_dice:.4f} - Recall: {mean_recall:.4f} - Precision: {mean_precision:.4f}")
    print(f"Average Inference Time: {avg_time:.4f} seconds - FPS: {fps:.2f}")
