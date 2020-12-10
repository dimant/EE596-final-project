import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class Abstract(object):
    labels = { "Arial": 0, "Times New Roman":1 }

    def __init__(self, fname, transform):
        im = cv2.imread(fname)
        self.image = torch.from_numpy(im).permute(2, 0, 1).float()

        g = re.search(".+(\d+) ([a-zA-Z0-9 ]+).png", fname)

        self.font = g.group(2)

    def getLabel(self):
        l = self.labels[self.font]

        return l

    def getFont(self):
        return self.font

    def getImage(self):
        return self.image

class AbstractsDataset(Dataset):
    def __init__(self, directory, transform):
        self.entries = []
        self.directory = directory
        self.transform = transform

        for entry in os.scandir(self.directory):
            if entry.path.endswith(".png") and entry.is_file():
                self.entries.append(Abstract(entry.path, transform))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        label = entry.getLabel()
        image = entry.getImage()

        image = self.transform(image)

        return image, label

def test():
    d = AbstractsDataset("data-1")
    dataloader = DataLoader(d, batch_size=4, shuffle=True, num_workers=1)
    
    for data in dataloader:
        images, labels = data

        print(labels)

# if __name__ == '__main__':
#     test()