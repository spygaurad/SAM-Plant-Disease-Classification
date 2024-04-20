import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, ToTensor, RandomRotation, RandomAffine, Resize, Normalize
from PIL import Image
from sklearn.model_selection import train_test_split
import torch

'''
#Dataloader for PlantVillage Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        with open(csv_file, 'r') as f:
            for row in f:
                file_path, label = row.split(',')
                label = label.strip()  
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                label_idx = self.class_to_idx[label]
                self.data.append((file_path, label_idx))

    def __getitem__(self, index):
        file_path, label = self.data[index]
        try:
            image = Image.open(file_path).convert('RGB')
        except:
            return None, label
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)



def get_dataloader(root_dir, batch_size):

    train_transform = transforms.Compose([
        RandomRotation(degrees=45), 
        RandomAffine(degrees=0, shear=10),  # Random skewness and shear up to 10 degrees
        Resize((224, 224)),  # Resize to 224x224
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(os.path.join(root_dir, "train.csv"), transform=train_transform)
    val_dataset = CustomDataset(os.path.join(root_dir, "valid.csv"), transform=val_transform)  # Resize validation images to 224x224
    test_dataset = CustomDataset(os.path.join(root_dir, "test.csv"), transform=val_transform)  # Resize validation images to 224x224

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # for element in train_loader:
    #     print(element)
    return train_loader, val_loader, test_loader



# train_data, val_data, test_data = get_dataloader("Dataset/Plant_Village/", batch_size=4)
'''




#Dataloader for New Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, classes, transform=None):
        self.transform = transform
        self.data = []
        self.classes = classes
        self.inverted_classes = {v: k for k, v in self.classes.items()}
        with open(csv_file, 'r') as f:
            next(f)
            for row in f:
                file_path, label = row.split(',')
                label = label.strip()  
                label = [cls_label for cls_label in self.classes.values() if label in cls_label]
                if label:
                    label = label[0]
                    label_idx = self.inverted_classes[label]
                    # Convert the label to a one-hot encoded vector
                    one_hot_label = torch.zeros(len(classes))
                    one_hot_label[label_idx] = 1
                    self.data.append((file_path, one_hot_label))
                else:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]

        image = Image.open(file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(batch_size, classes):

    transform = transforms.Compose([
        RandomRotation(degrees=45), 
        RandomAffine(degrees=0, shear=10),  # Random skewness and shear up to 10 degrees
        Resize((224, 224)),  # Resize to 224x224
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    custom_dataset = CustomDataset(csv_file='cropped_images.csv', classes=classes, transform=transform)
    train_dataset, test_dataset = train_test_split(custom_dataset, test_size=0.1, random_state=42)

    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_data_loader, test_data_loader