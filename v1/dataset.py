import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AlphaBeta(Dataset):
    def __init__(self, data_path, is_training=True, transform=None):
        self.data_path = data_path
        self.is_training = is_training
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.file = os.path.join(self.data_path, "train.txt" if self.is_training else "test.txt")
        with open(self.file, "r") as f:
            self.files = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        label = file.split("_")[0]
        label = torch.tensor(ord(label) - ord("a"))
        image = cv2.imread(os.path.join(self.data_path, "data", file), cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    dataset = AlphaBeta("dataset")
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][0].min(), dataset[0][0].max(), dataset[0][0].dtype)
    print(dataset[0][1])
