# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import random
from model_structure import get_preprocessing_transforms,BCNN, train_model, evaluate_model, get_resnet_model, get_squ_model

# %%
seed = 441
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
MODEL_NUMBER = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
INPUT_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Get transforms
train_transform, val_transform = get_preprocessing_transforms(INPUT_SIZE)

# %%
# Set data set directory
train_dir = f'data4model_{MODEL_NUMBER}/train/'
val_dir = f'data4model_{MODEL_NUMBER}/test/'

# %%
# Create datasets
train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

# %%
# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

# %%
model = get_resnet_model(2).to(DEVICE)
criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.eval()

# %%
summary(model, (3, INPUT_SIZE, INPUT_SIZE))

# %%
tracker = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, MODEL_NUMBER)


