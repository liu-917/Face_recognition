import cv2
import os
from PIL import Image
import random
import numpy as np
import shutil
def yuchuli(dataset_dir, output_dir):

    input_size = (224, 224)


    rotation_range = 20
    width_shift_range = 0.1
    height_shift_range = 0.1


    os.makedirs(output_dir, exist_ok=True)


    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):

            output_label_dir = os.path.join(output_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)


            for file_name in os.listdir(label_dir):

                image_path = os.path.join(label_dir, file_name)
                image = Image.open(image_path)


                output_path = os.path.join(output_label_dir, file_name)

                image = image.resize(input_size)
                Image.fromarray((np.array(image) / 255.0 * 255).astype(np.uint8)).save(output_path)


                angle = random.uniform(-rotation_range, rotation_range)
                image = image.rotate(angle)


                width_shift = random.uniform(-width_shift_range, width_shift_range) * input_size[0]
                height_shift = random.uniform(-height_shift_range, height_shift_range) * input_size[1]
                image = image.transform(input_size, Image.AFFINE, (1, 0, width_shift, 0, 1, height_shift))


                image = np.array(image) / 255.0

                # print(output_path)
                file_name1='_'+file_name
                output_path1=os.path.join(output_label_dir, file_name1)
                Image.fromarray((image * 255).astype(np.uint8)).save(output_path1)


dataset_dir = r"./photo/data"
output_dir = r"./photo/new_data"
yuchuli(dataset_dir, output_dir)


def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):

    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)


    files = os.listdir(source_dir)
    random.shuffle(files)


    num_files = len(files)
    train_split = int(num_files * split_ratio[0])
    val_split = int(num_files * (split_ratio[0] + split_ratio[1]))


    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]


    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))


name=os.listdir(r'./photo/data')
for i in name:
    source_dir = fr"./photo/new_data/{i}"
    train_data_dir = fr"./photo/new_data/train_data_dir/{i}"
    val_data_dir = fr"./photo/new_data/val_data_dir/{i}"
    test_data_dir = fr"./photo/new_data/test_data_dir/{i}"

    split_data(source_dir, train_data_dir, val_data_dir, test_data_dir)

name = os.listdir(r'./photo/data')
for i in name:
    repath = rf"./photo/new_data/{i}"
    if repath:
        shutil.rmtree(repath)