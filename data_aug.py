
import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import create_dir
from data import load_data

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Crop,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RGBShift,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    size = (512, 512)

    for idx, (x_path, y_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        image_name = os.path.splitext(os.path.basename(x_path))[0]
        mask_name = os.path.splitext(os.path.basename(y_path))[0]

        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        y = cv2.imread(y_path, cv2.IMREAD_COLOR)

        x = cv2.resize(x, size)
        y = cv2.resize(y, size)

        if augment:
            aug_x = [x]
            aug_y = [y]

            aug = Crop(p=1, x_min=0, x_max=size[0], y_min=0, y_max=size[1])
            augmented = aug(image=x, mask=y)
            aug_x.append(augmented["image"])
            aug_y.append(augmented["mask"])

            for transformation in [
                RandomRotate90(p=1),
                ElasticTransform(p=1, alpha=120, sigma=120 * 0.05),
                GridDistortion(p=1),
                OpticalDistortion(p=1, distort_limit=2),
                VerticalFlip(p=1),
                HorizontalFlip(p=1),
                RGBShift(p=1),
                ChannelShuffle(p=1),
                CoarseDropout(p=1, num_holes_range=(1, 10), hole_height_range=(16, 32), hole_width_range=(16, 32), fill=0),
                GaussNoise(p=1)
            ]:
                augmented = transformation(image=x, mask=y)
                aug_x.append(augmented["image"])
                aug_y.append(augmented["mask"])

            # Grayscale
            gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
            aug_x.append(gray)
            aug_y.append(y)

        else:
            aug_x = [x]
            aug_y = [y]

        for i, m in zip(aug_x, aug_y):
            tmp_image_name = f"{image_name}_{idx}.jpg"
            tmp_mask_name  = f"{mask_name}_{idx}.jpg"

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path  = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


def main():
    np.random.seed(42)

    dataset_path = ""
    (train_x, train_y), (val_x, val_y), _ = load_data(dataset_path)

    print("Train: ", len(train_x))
    print("Valid: ", len(val_x))

    augmented_path = ""
    create_dir(os.path.join(augmented_path, "train", "images"))
    create_dir(os.path.join(augmented_path, "train", "masks"))
    create_dir(os.path.join(augmented_path, "val", "images"))
    create_dir(os.path.join(augmented_path, "val", "masks"))
    create_dir(os.path.join(augmented_path, "test", "images"))
    create_dir(os.path.join(augmented_path, "test", "masks"))


    augment_data(train_x, train_y, os.path.join(augmented_path, "train"))
    augment_data(val_x, val_y, os.path.join(augmented_path, "val"))

if __name__ == "__main__":
    main()
