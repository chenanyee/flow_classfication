import os
import pandas as pd

test_dir = './dataset_cnn/train_ori/NH'

# 初始化列表存储文件路径和标签
file_paths = []
labels = []

# 遍历目录中的文件
for filename in os.listdir(test_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理图片文件
        file_path = os.path.join(test_dir, filename)
        label = filename[0]  # 文件名的第一个字母作为标签
        if label in ['N', 'H']:  # 只处理标签为N或H的文件
         if label == 'N':  # 将N标签改为NH
          label = 'NH'
        file_paths.append(file_path)
        labels.append(label)

# 创建DataFrame
df = pd.DataFrame({
    'path': file_paths,
    'label': labels
})

# 保存为CSV文件
df.to_csv('./dataset_cnn/train_1.csv', index=False)

print("测试集CSV文件已生成")

