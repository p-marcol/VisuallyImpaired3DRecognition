import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, length=100, num_classes=4, img_size=224):
        self.length = length
        self.num_classes = num_classes
        self.img_size = img_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Generate a random image tensor and a random label
        image = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label