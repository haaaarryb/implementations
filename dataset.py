import os
from torch.utils.data import Dataset
from PIL import Image

class ImgDataset(Dataset):

    def __init__(self, root_dir='imgs', transform=None):
        self.root_dir = root_dir
        self.img_fps = os.listdir(root_dir)
        self.transform = transform

    def __getitem__(self, idx):
        img_fp = os.path.join(self.root_dir, self.img_fps[idx])
        img = Image.open(img_fp)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_fps)



