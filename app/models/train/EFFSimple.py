
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import multiprocessing as mp
import torchvision
import timm


from matplotlib.pyplot import imshow
from PIL import Image
from tempfile import TemporaryDirectory
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.transforms.functional import InterpolationMode
from transformers import get_cosine_schedule_with_warmup



# OS ENV SETUP

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CONFIG

TRAIN_DIR = 'X:\\REpoS\\Yogi Eye\\data\\Ayurved.v2\\train'
VAL_DIR = 'X:\\REpoS\\Yogi Eye\\data\\Ayurved.v2\valid'
TEST_DIR = 'X:\\REpoS\\Yogi Eye\\data\\Ayurved.v2\\test'
DATA_DIR = 'X:\\REpoS\\Yogi Eye\\data\\Ayurved.v2'
SUB_DIR = 0
SEED = 279
TRAIN_BS = 7
TEST_BS = 100
NUM_CLASSES = 128
EMBEDDING_SIZE = 1280
NUM_EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0
WARMUP_EPOCHS = 0
LOGGING_INTERVAL = 100
N_CORES = mp.cpu_count()



if torch.cuda.is_available():
    DEVICE = torch.device(type='cuda')
else:
    DEVICE = torch.device('cpu')

print(f'USING device: {DEVICE}')



def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



set_seed()


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



def convert_4_channel_to_3_channel(image):
    """
    Convert 4-channel RGBA image to 3-channel RGB image
    """
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image


data_transforms = {
    'train':transforms.Compose([
    transforms.Lambda(convert_4_channel_to_3_channel),
    transforms.Resize(
        size=(250, 250), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomRotation(degrees=(-180, 180)),
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.7, 1.3)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]),
    'valid':transforms.Compose([
    transforms.Lambda(convert_4_channel_to_3_channel),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor,
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    }


img_dataset = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'valid']}


dataloader ={x:DataLoader(img_dataset[x], batch_size=TRAIN_BS, drop_last=True, shuffle=True, num_workers=N_CORES) for x in ['train','valid']}


dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'valid']}
class_names = img_dataset['train'].classes


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



set_seed(SEED)
model = EffNet(num_classes=NUM_CLASSES, embedding_size=EMBEDDING_SIZE)
model.to(DEVICE)

# with open('model_state.pt', 'wb') as f: 
#         torch.save(model().state_dict(), f)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARMUP_EPOCHS,
    num_training_steps=dataset_sizes['train']*NUM_EPOCHS
)

scaler = GradScaler()


def train_model(model,  optimizer, scaler, scheduler, num_epochs):
    
    since = time.time()

    
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'model_state.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval()   

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with autocast():
                            logits = model(inputs)
                            _, preds = torch.max(logits, 1)
                            loss = F.cross_entropy(logits, labels, reduction='mean')
                        
                        

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model




def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader['val']):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_tr = train_model(model,optimizer,scaler, scheduler,NUM_EPOCHS)


visualize_model(model_tr)



