import os
from cat_model import cat_model
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from urllib.request import urlopen
from train import train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from image_dataset import image_dataset

if __name__ == '__main__':
    inputSize = 1200
    dataTransformsTrain = transforms.Compose([
        transforms.RandomResizedCrop(inputSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataTransformsValid = transforms.Compose([
        transforms.Resize(inputSize),
        transforms.CenterCrop(inputSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainDatasets = image_dataset("dataset/train.csv", "dataset/cosmetics-all", dataTransformsTrain)
    validDatasets = image_dataset("dataset/valid.csv", "dataset/cosmetics-all", dataTransformsValid)
    classList = "\n".join(trainDatasets.__class__())

    # make classes.txt
    with open("dataset/classes.txt", "w") as f:
        f.write(classList)

    dataloadersTrain = torch.utils.data.DataLoader(trainDatasets, batch_size=32, shuffle=True, num_workers=4)
    dataloadersValid = torch.utils.data.DataLoader(validDatasets, batch_size=32, shuffle=False)

    # load model
    model = cat_model(numClasses=7).to(device)

    # set optimization function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)

    # training
    model_ft = train(model, dataloadersTrain, dataloadersValid, optimizer, criterion, 11)

