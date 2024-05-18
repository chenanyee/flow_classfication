import os
import shutil
import pandas as pd
import random

# 原始图片文件夹路径
#h_folder = './dataset/H/'
nh_folder = './dataset_cnn/hotspot'

# 目标文件夹路径
train_folder_nh = './dataset_cnn/train/'
valid_folder_nh = './dataset_cnn/valid/'

# 创建新文件夹
os.makedirs(train_folder_nh, exist_ok=True)
os.makedirs(valid_folder_nh, exist_ok=True)

# 获取H和NH文件夹中的所有文件
nh_files = os.listdir(nh_folder)

# 打乱文件列表
#random.shuffle(h_files)
random.shuffle(nh_files)

# 分割比例
split_ratio = 0.8
#num_train_h = int(len(h_files) * split_ratio)
num_train_nh = int(len(nh_files) * split_ratio)

# 移动图片并生成CSV文件
def move_and_create_csv(files, num_train, train_folder, valid_folder, folder, label):
    train_paths = []
    valid_paths = []
    
    for i, file_name in enumerate(files):
        src_path = os.path.join(folder, file_name)
        if i < num_train:
            shutil.copy(src_path, train_folder)
            train_paths.append({'path': os.path.join(train_folder, file_name), 'label': label})
        else:
            shutil.copy(src_path, valid_folder)
            valid_paths.append({'path': os.path.join(valid_folder, file_name), 'label': label})

    df_train = pd.DataFrame(train_paths)
    df_valid = pd.DataFrame(valid_paths)

    df_train.to_csv(os.path.join(train_folder, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(valid_folder, 'valid.csv'), index=False)

move_and_create_csv(nh_files, num_train_nh, train_folder_nh, valid_folder_nh, nh_folder, 'NH')

