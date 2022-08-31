import pandas as pd
import torchvision
from torch.utils.data import Dataset

class FashionMNISTDataset(Dataset):

    def __init__(self, img_dir, annotation_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_name = self.img_labels["file_name"].iloc[idx]
        file_path = "".join([self.img_dir, file_name])
        img = torchvision.io.read_image(file_path)
        label = self.img_labels["label"].iloc[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label