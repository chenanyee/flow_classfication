import os
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset


class image_dataset(Dataset):
    def __init__(self, csvFile, rootPath, transform):
        df = pd.read_csv(csvFile)
        self.rootPath = rootPath
        self.xTrain = df['path']
        #label_mapping = {'NH': 0, 'H': 1}
        #df['label'] = df['label'].map(label_mapping)
        #self.yTrain = df['label']
        #self.yTrain = pd.factorize(df['label'], sort=True)[0]
        self.yTrain = df['label'].map({'NH': 0, 'H': 1}).astype(int)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootPath, self.xTrain[index]))
        img = img.convert('1')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)

