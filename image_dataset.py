import os
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset


class image_dataset(Dataset):
    def __init__(self, csvFile, rootPath, transform):
        df = pd.read_csv(csvFile)
        self.rootPath = rootPath
        self.xTrain = df['path']
        self.yTrain = pd.factorize(df['label'], sort=True)[0]
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rootPath, self.xTrain[index]))
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.yTrain[index]

    def __len__(self):
        return len(self.xTrain.index)

