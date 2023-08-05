##Importing Libraries

import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
from tqdm.auto import tqdm
import requests
import zipfile
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

#Data preparation and augmentation
data_path = Path("data/")
img_path = data_path / "food"

# If the image folder doesn't exist, download it and prepare it...

if img_path.is_dir():
    print(f"{img_path} directory exists.")
else:
    print(f"Did not find {img_path} directory, creating one...")
    img_path.mkdir(parents=True, exist_ok=True)

    # Download the data
    with open(data_path / "data.zip", "wb") as f:
        request = requests.get("https://github.com/ArtificialT800/FoodModel/raw/main/food_101.zip")
        print("Downloading dataset...")
        f.write(request.content)

    # Unzip the data
    with zipfile.ZipFile(data_path / "data.zip", "r") as zip_ref:
        print("Unzipping data...")
        zip_ref.extractall(img_path)

## Creating splits

train_dir = img_path / "train"
test_dir = img_path / "test"


#Data augmentation

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

###Creating Dataset
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None,
                                  )

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 target_transform=None)

print(train_data, test_data)

##Creating the train and test dataloaders

BATCH_SIZE = 16 
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print(len(train_dataloader), len(test_dataloader))

class_names = train_data.classes
