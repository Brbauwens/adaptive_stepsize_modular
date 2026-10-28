import os
from datasets import load_dataset, Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Transforms (Data Augmentation for Train, Clean Resize for Val)
imNet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imNet_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess functions for the transforms
def imNet_preprocess_train(examples):
    examples["pixel_values"] = [imNet_train_transform(img.convert("RGB")) for img in examples["image"]]
    return examples

def imNet_preprocess_val(examples):
    examples["pixel_values"] = [imNet_val_transform(img.convert("RGB")) for img in examples["image"]]
    return examples

# Collate function to stack items into standard PyTorch batches
def im100_collate_fn(batch):
    images = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return images, labels

def _load_ImageNet(data_dir, batch_size, num_workers, image_convert = False):
    # 1. Point to your local data folder
    # Automatically group parquet files into train and validation splits based on filenames
    all_files = os.listdir(data_dir)
    train_files = [os.path.join(data_dir, f) for f in all_files if f.endswith(".parquet") and "train" in f.lower()]
    val_files = [os.path.join(data_dir, f) for f in all_files if f.endswith(".parquet") and ("val" in f.lower() or "test" in f.lower())]

    # 2. Load the splits together
    data_files = {"train": train_files, "validation": val_files}
    dataset = load_dataset("parquet", data_files=data_files)
    if (image_convert):
        dataset = dataset.cast_column("image", Image())

    # Apply separate named transformations to each split (safely picklable)
    dataset["train"].set_transform(imNet_preprocess_train)
    dataset["validation"].set_transform(imNet_preprocess_val)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=im100_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset["validation"], 
        batch_size=batch_size,
        shuffle=False,
        collate_fn=im100_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

#Procedure is used to load data in parquet-format, clonned from https://huggingface.co/datasets/clane9/imagenet-100
#If lib `datasets` is absent locally, install with command: `pip install datasets`
def load_ImageNet100():
    DATA_DIR = "/opt/software/datasets/imagenet-100/data/"
    BATCH_SIZE = 32
    NUM_WORKERS = 24 #Equal to CPU physical core count
    return _load_ImageNet(DATA_DIR, BATCH_SIZE, NUM_WORKERS)

#Procedure is used to load data in parquet-format, clonned from https://huggingface.co/datasets/zh-plus/tiny-imagenet
#If lib `datasets` is absent locally, install with command: `pip install datasets`
def load_ImageNetTiny():
    DATA_DIR = "/opt/software/datasets/tiny-imagenet/data/"
    BATCH_SIZE = 32
    NUM_WORKERS = 24 #Equal to CPU physical core count
    return _load_ImageNet(DATA_DIR, BATCH_SIZE, NUM_WORKERS, image_convert = True)
