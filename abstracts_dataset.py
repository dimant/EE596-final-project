import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

def extract_font(fname):
    g = re.search(".+(\d+) ([a-zA-Z0-9 ]+).png", fname)
    return g.group(2)

class Abstract(object):

    def __init__(self, fonts, fname, transform):
        im = cv2.imread(fname)
        # crop top line. The reason is that a lot of the input data only has one line
        # if we do random crop we will end up with a lot of just white data
        im = im[:32, :128]
        self.image = torch.from_numpy(im).permute(2, 0, 1).float()
        self.font = extract_font(fname)
        self.label = fonts.index(self.font)

    def getLabel(self):
        return self.label

    def getFont(self):
        return self.font

    def getImage(self):
        return self.image

def contains_any(string, fonts):
    actual = extract_font(string)

    return actual in fonts

class AbstractsDataset(Dataset):

    def __init__(self, directory, transform, fonts, max_count=0):
        self.entries = []
        self.directory = directory
        self.transform = transform

        i = 0

        for entry in os.scandir(self.directory):
            fname = entry.path

            if fname.endswith(".png") and entry.is_file() and contains_any(fname, fonts):
                self.entries.append(Abstract(fonts, fname, transform))

            if max_count > 0 and i < max_count - 1:
                i += 1
            else:
                break

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