from dataloader import get_dataloader
from network import EfficientNet, DomainAdaptiveNet
from gradhandler import GradientReversalLayer

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from PIL import Image, ImageDraw
from PIL import Image
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import csv
import matplotlib.pyplot as plt


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
BATCH_SIZE = 2
MODEL_NAME = "TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V3_Test-Training_SmallSamples"



class Model():

    def __init__(self, trained=False):
        self.model = EfficientNet()
        # self.model = DomainAdaptiveNet().to(DEVICE)
        # self.shared_layers = self.model.effnet
        # self.classificationLayer = self.model.classificationLayer
        # self.domainClassificationLayer = self.model.domainClassificationLayer
        # self.model.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_5_140.pth', map_location=torch.device(DEVICE)))
        if trained: self.model.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_6_10.pth', map_location=torch.device(DEVICE)))

        self.classes = {
            0: "Tomato Bacterial Spot", 
            1: "Tomato Early Blight", 
            2: "Tomato Healthy", 
            3: "Tomato Late Blight", 
            4: "Tomato Leaf Mold", 
            5: "Tomato Septoria Leaf Spot", 
            6: "Tomato Spider Mites", 
            7: "Tomato Target Spot", 
            8: "Tomato Mosaic Virus",
            9: "Tomato Yellow Leaf Curl Virus",
        }

        # self.classes = { 
        #     0: "Tomato Early Blight",
        #     1: "Tomato Late Blight",
        #     2: "Tomato Septoria Leaf Spot", 
        # }



    def train(self, dataset, loss_func, optimizer, lambda_val):
        self.model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_domain_loss = 0.0
        counter = 0

        for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
            counter += 1
            optimizer.zero_grad()
            image, label = img.to(DEVICE), label.to(DEVICE)


            #Code for training the efficientnet model
            # '''
            output = self.model(image)
            loss = loss_func(output, label)
            running_loss += loss.item()
            # '''
       
            #code for training the domain adversarial neural network
            '''
            # Forward pass through shared layers
            shared_features = self.shared_layers(image)

            # Classification branch
            diseaseClass = self.classificationLayer(shared_features)
            loss_cls = loss_func(diseaseClass, label)
            running_loss += loss_cls.item()

            # Domain classification branch
            domain_labels = torch.zeros(image.size(0), dtype=torch.long).to(DEVICE)  # Assume source domain
            reversed_features = GradientReversalLayer.apply(shared_features, lambda_val)
            domainClass = self.domainClassificationLayer(reversed_features)
            loss_domain = loss_func(domainClass, domain_labels)
            running_domain_loss += loss_domain.item()

            # Total loss
            loss = loss_cls + loss_domain
            '''

            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(1)
            correct = pred == label.argmax(1)
            running_correct += correct.sum().item()

        # Loss and accuracy for a complete epoch
        epoch_loss = running_loss / (counter * BATCH_SIZE)
        epoch_acc = 100. * (running_correct / (counter * BATCH_SIZE))

        return epoch_loss, epoch_acc




    def validate(self, dataset):

        self.model.eval()
        running_correct = 0.0
        counter = 0

        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                img, label = img.to(DEVICE), label.to(DEVICE)
                outputs = self.model(img)

                #calculate accuracy
                pred = outputs.argmax(1)
                correct = pred == label
                running_correct += correct.sum().item()

        # loss and accuracy for a complete epoch
        epoch_acc = 100. * (running_correct / (counter*BATCH_SIZE))
        return epoch_acc



    def test(self, dataset):

        # self.model.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_3_200.pth'))
        running_correct = 0.0
        counter = 0

        # num = random.randint(0, len(dataset)-1)
        self.model.eval()
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                img, label = img.to(DEVICE), label.to(DEVICE)
                outputs = self.model(img)
                #calculate accuracy
                pred = outputs.argmax(1)
                correct = pred == label
                running_correct += correct.sum().item()

        # loss and accuracy for a complete epoch
        epoch_acc = 100. * (running_correct / (counter))
    
        return epoch_acc


 
    def fit(self, epochs, lr):

        print(f"Using {DEVICE} device...")
        print("Loading Datasets...")
        train_data, test_data = get_dataloader(BATCH_SIZE, self.classes)
        # print(f"Training Samples: {len(train_data)*BATCH_SIZE}\nValidation Samples: {len(val_data)*BATCH_SIZE}\nTesting Samples: {len(test_data)*BATCH_SIZE}")
        print(f"Training Samples: {len(train_data)*BATCH_SIZE}\nTesting Samples: {len(test_data)*BATCH_SIZE}")
        print("Dataset Loaded.")
        print("Initializing Parameters...")
        self.model = self.model.to(DEVICE)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"The total parameters of the model are: {total_params}")
        print(f"Initializing the Optimizer")
        optimizer = optim.AdamW(self.model.parameters(), lr)
        print(f"Beginning to train...")

        crossEntropyLoss = nn.CrossEntropyLoss()
        # train_loss_epochs, val_acc_epochs, test_acc_epochs = [], [], []
        train_loss_epochs, test_acc_epochs = [], []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')
        os.makedirs("checkpoints/", exist_ok=True)
        os.makedirs("saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):
 
            print(f"Epoch No: {epoch}")
            train_loss, train_acc = self.train(dataset=train_data, loss_func=crossEntropyLoss, optimizer=optimizer, lambda_val=0.1)
            # val_acc = self.validate(dataset=val_data)
            test_acc = self.test(dataset=test_data)
            train_loss_epochs.append(train_loss)
            # val_acc_epochs.append(val_acc)
            test_acc_epochs.append(test_acc)
            # print(f"Train Loss:{train_loss}, Train Accuracy:{train_acc}, Validation Accuracy:{val_acc}, Test Accuracy: {test_acc}")
            print(f"Train Loss:{train_loss}, Train Accuracy:{train_acc}, Test Accuracy: {test_acc}")


            if max(test_acc_epochs) == test_acc:
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                }, f"checkpoints/{MODEL_NAME}.tar")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            # writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            
            
            print("Saving model")
            torch.save(self.model.state_dict(), f"saved_model/{MODEL_NAME}_{epoch}.pth")
            print("Model Saved")
    
            print("Epoch Completed. Proceeding to next epoch...")


        print(f"Training Completed for {epochs} epochs.")


    def infer_a_random_sample(self):
        
        try:
            os.makedirs(f"test_samples/{MODEL_NAME}", exist_ok=True)
        except:
            pass
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        with open('Dataset/Plant_Village/test.csv', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = list(csvreader)
            random_row = random.choice(rows)
            path = random_row[0]
            label = random_row[1]

            image = Image.open(path)
            imageT = transform(image).unsqueeze(0).to(DEVICE)
            outputs = self.model(imageT)
            pred = outputs.argmax(1)
            pred_label = self.classes[pred.item()]
            print(pred_label)
            print(label)


            draw = ImageDraw.Draw(image)
            draw.text((image.width - 200, 0), f"Real: {label}", fill='red')
            draw.text((image.width - 200, 20), f"Predicted: {pred_label}", fill='blue')
            image.save(f"test_samples/{MODEL_NAME}/{label} -> {pred_label}.jpg")
            print("Saved a sample")




    def infer_a_sample(self, image):

        image = image.to(DEVICE)
        self.model.eval()
        # Forward pass the image through the model.
        prediction = nn.Softmax(dim=1)(self.model(image)).max(1)
        class_prob, class_index = round(prediction.values.item(), 3), prediction.indices.item()
        class_name = self.classes[class_index]
        return f'{class_name}: {class_prob*100}%'
        




model = Model(trained=True)
# model.fit(20, 1e-6)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a list to store the results
results = []

# Load the CSV file
csv_file = "filtered_images.csv"
output_folder = "outputs/bg_output"
os.makedirs(output_folder, exist_ok=True)
# Open the CSV file and read the rows
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        file_path, label = row
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        prediction = model.infer_a_sample(image.to(DEVICE))
        image = transforms.ToPILImage()(image.squeeze(0).cpu())
        draw = ImageDraw.Draw(image)
        text_width, text_height = draw.textsize(prediction)
        x = (image.width - text_width) // 2
        y = 10
        draw.text((x, y), prediction, fill="white")
        x = (image.width - len(label) * 10) // 2
        y = image.height - 30
        draw.text((x, y), label, fill="white")
        labeled_image_path = os.path.join(output_folder, fr"labeled_{file_path.split('/')[-1]}")
        image.save(labeled_image_path)
        print(f"Image: {file_path}, Prediction: {prediction}, Ground Truth: {label}")
        results.append([file_path, prediction, label])
# for i in range(10):
#     model.infer_a_random_sample()

output_csv_file = "outputs/bg_output/bg_predicted.csv"
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Path", "Prediction", "Ground Truth"])
    writer.writerows(results)