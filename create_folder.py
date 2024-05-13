import os
import shutil
import pandas as pd

# 原始图片文件夹路径
h_folder = './dataset/H/'
nh_folder = './dataset/NH/'

# 目标文件夹路径
train_folder_h = './dataset/train/H/'
train_folder_nh = './dataset/train/NH/'
valid_folder_h = './dataset/valid/H/'
valid_folder_nh = './dataset/valid/NH/'

# 创建新文件夹
os.makedirs(train_folder_h, exist_ok=True)
os.makedirs(train_folder_nh, exist_ok=True)
os.makedirs(valid_folder_h, exist_ok=True)
os.makedirs(valid_folder_nh, exist_ok=True)

# 获取H和NH文件夹中的所有文件
h_files = os.listdir(h_folder)
nh_files = os.listdir(nh_folder)

# 分割比例
split_ratio = 0.7
num_train_h = int(len(h_files) * split_ratio)
num_train_nh = int(len(nh_files) * split_ratio)


# 移动图片并生成CSV文件
def move_and_create_csv(files, train_folder, valid_folder, label):
    for i, file_name in enumerate(files):
        if i < num_train_h:
            shutil.copy(os.path.join(h_folder, file_name), train_folder)
        else:
            shutil.copy(os.path.join(h_folder, file_name), valid_folder)




def move_and_create_csv_nh(files, train_folder, valid_folder, label):
    for i, file_name in enumerate(files):
        if i < num_train_nh:
            shutil.copy(os.path.join(nh_folder, file_name), train_folder)
        else:
            shutil.copy(os.path.join(nh_folder, file_name), valid_folder)

    df_train = pd.DataFrame({'name': os.listdir(train_folder), 'label': label})
    df_valid = pd.DataFrame({'name': os.listdir(valid_folder), 'label': label})

    df_train.to_csv(train_folder + 'train.csv', index=False)
    df_valid.to_csv(valid_folder + 'valid.csv', index=False)




move_and_create_csv(h_files, train_folder_h, valid_folder_h, 'H')
move_and_create_csv_nh(nh_files, train_folder_nh, valid_folder_nh, 'NH')

