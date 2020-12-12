import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

def extract_font(fname):
    g = re.search(".+(\d+) ([a-zA-Z0-9 ]+).png", fname)
    return g.group(2)

class Abstract(object):
    labels = { 
        "Arial": 0,
        "Times New Roman": 1,
        "Comic Sans": 2,
        "Courier New": 3,
        "Calibri": 4,
        "Candara": 5,
        "Consolas": 6,
        "Georgia": 7,
        "Corbel": 8,
        "Arial Black": 9
        }

    def __init__(self, fname, transform):
        im = cv2.imread(fname)
        self.image = torch.from_numpy(im).permute(2, 0, 1).float()
        self.font = extract_font(fname)

    def getLabel(self):
        l = self.labels[self.font]

        return l

    def getFont(self):
        return self.font

    def getImage(self):
        return self.image

def contains_any(string, fonts):
    actual = extract_font(string)

    return actual in fonts

class AbstractsDataset(Dataset):
    def __init__(self, directory, transform, fonts=Abstract.labels):
        self.entries = []
        self.directory = directory
        self.transform = transform

        for entry in os.scandir(self.directory):
            fname = entry.path

            if fname.endswith(".png") and entry.is_file() and contains_any(fname, fonts):
                self.entries.append(Abstract(fname, transform))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        label = entry.getLabel()
        image = entry.getImage()

        image = self.transform(image)

        return image, label

def transform_identity(obj):
    return obj

def test():
    d = AbstractsDataset("data-1", transform_identity, ["Arial"])
    dataloader = DataLoader(d, batch_size=4, shuffle=True, num_workers=1)
    
    for data in dataloader:
        images, labels = data

        print(labels)

# if __name__ == '__main__':
#     test()