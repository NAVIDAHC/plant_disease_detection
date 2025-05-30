import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_dual_datasets(dataset_path1, dataset_path2, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    ds1 = datasets.ImageFolder(dataset_path1, transform=transform)
    ds2 = datasets.ImageFolder(dataset_path2, transform=transform)

    loader1 = DataLoader(ds1, batch_size=batch_size, shuffle=False)
    loader2 = DataLoader(ds2, batch_size=batch_size, shuffle=False)

    return loader1, loader2

def load_split_train_val(dataset_path, split_ratio=0.3, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=split_ratio, stratify=[dataset.targets[i] for i in indices])
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

