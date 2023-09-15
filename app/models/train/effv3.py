# imports

import os
import time

import pandas as pd
from PIL import Image
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import get_cosine_schedule_with_warmup
import timm

# OS ENV SETUP

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CONFIG

TRAIN_DIR = 'data/indo_herb/train'
TEST_DIR = 'data/indo_herb/test'
SUB_DIR = 0
SEED = 279
TRAIN_BS = 7
TEST_BS = 80
NUM_CLASSES = 50
EMBEDDING_SIZE = 500
NUM_EPOCHS = 7
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0
WARMUP_EPOCHS = 0
LOGGING_INTERVAL = 100
N_CORES = mp.cpu_count()


if torch.cuda.is_available():
    DEVICE = torch.device(type='cuda')
else:
    DEVICE = torch.device('cpu')




def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed()


# CREATING DATA FRAME

class_label_names = os.listdir(TRAIN_DIR)


train_data = []
for label_index, label_name in enumerate(class_label_names):
    features = os.listdir(os.path.join(TRAIN_DIR, label_name))
    train_data.extend(
        [os.path.join(label_name, feature), label_index]
        for feature in features
    )

train_df = pd.DataFrame(train_data, columns=['file_path', 'label'])



# test file config

test_files = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'file_path': test_files})
test_df['label'] = 9999


# COnfiguring Data Set

class PlantDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.file_path = df['file_path']
        self.y = df['label']

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.file_path[index]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


# Augmentation of Image

def convert_4_channel_to_3_channel(image):
    """
    Convert 4-channel RGBA image to 3-channel RGB image
    """
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

# Transformation & Normalization


train_transforms = transforms.Compose([
    transforms.Lambda(convert_4_channel_to_3_channel),
    transforms.Resize(
        size=(250, 250), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.7, 1.3)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_transforms = transforms.Compose([
    transforms.Lambda(convert_4_channel_to_3_channel),
    transforms.Resize(
        size=(224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# apply above

train_dataset = PlantDataset(df=train_df,
                                    img_dir=TRAIN_DIR,
                                    transform=train_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=TRAIN_BS,
                          drop_last=True,
                          shuffle=True,
                          num_workers=N_CORES)


test_dataset = PlantDataset(df=test_df,
                                   img_dir=TEST_DIR,
                                   transform=test_transforms)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=TEST_BS,
                         drop_last=False,
                         shuffle=False,
                         num_workers=N_CORES)



# Model Initialization and Setup

class EffNet(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(EffNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.backbone = timm.create_model(
            'efficientnet_b1',
            pretrained=True,
            num_classes=self.num_classes
        )
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.embedding_size,256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, self.num_classes)
            )
        
    def forward(self,x):
        return self.backbone(x)



# Model Config

set_seed(SEED)
model = EffNet(num_classes=NUM_CLASSES, embedding_size=EMBEDDING_SIZE)
model.to(DEVICE)


optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARMUP_EPOCHS,
    num_training_steps=len(train_loader)*NUM_EPOCHS
)

scaler = GradScaler()


if __name__ == "__main__":
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        
        for batch_idx, (features, target) in enumerate(train_loader):
            features = features.to(DEVICE)
            target = target.to(DEVICE)

            with autocast():
                logits = model(features)
                loss = F.cross_entropy(logits, target, reduction='mean')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            
            if not batch_idx % LOGGING_INTERVAL:
                print(
                    f'Epoch: {epoch + 1}/{NUM_EPOCHS}'
                    f' | Batch: {batch_idx}/{len(train_loader)}'
                    f' | Loss: {loss:.4f}'
                )

                
    elapsed = (time.time() - start_time) / 60
    print(f'Total training time: {elapsed:.3f} min')
