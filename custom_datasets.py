import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset

class dataset_1(Dataset):
    def __init__(self,target,transform=None, target_transform=None):
        path=open('path.txt',"r+").readline()
        self.img_labels=pd.read_csv(f"{path}\\archive1\\dataset_v2\\{target}\\label.csv")
        self.img_dir=f"{path}\\archive1\\dataset_v2\\{target}\\images"
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, id):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[id, 0])
        image = decode_image(img_path).float()/255.0
        label = int(self.img_labels.iloc[id, 1])
        x_center =self.img_labels.iloc[id,2]
        y_center =self.img_labels.iloc[id,3]
        width = self.img_labels.iloc[id,4]
        height = self.img_labels.iloc[id,5]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label,x_center,y_center,width,height
