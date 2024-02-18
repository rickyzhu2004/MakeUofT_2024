import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.datasets import ImageFolder 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import pandas as pd
import os
import csv
import random

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current working directory:", os.getcwd())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available! Training on GPU...')
else:
    device = torch.device("cpu")
    print('CUDA is not available. Training on CPU...')


#%% 
#Loading and split data 
data_path = r"C:\Users\Ricky\Desktop\MakeUoft_2024"
train_path = data_path + r"\train"
valid_path = data_path + r"\valid"
test_path = data_path + r"\test"
diseases = os.listdir(train_path)

num_classes = len(diseases)

train_set = ImageFolder(train_path, transform=transforms.ToTensor())
valid_set = ImageFolder(valid_path, transform=transforms.ToTensor()) 
test_set = ImageFolder(test_path, transform=transforms.ToTensor()) 

'''
num_samples = 76

indices = list(range(num_samples))
train_subset = torch.utils.data.Subset(train_set, indices)
val_subset = torch.utils.data.Subset(valid_set, indices)
test_subset = torch.utils.data.Subset(test_set, indices)
'''

class_names = train_set.classes

# Get the total number of classes
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Class names:", class_names)

#%%
#Model: transfer learning using resnet 152 
class ResNet(nn.Module):
    def __init__(self,num_classes):
        super(ResNet, self).__init__()
        self.name = "resnetclassifier"

        # Load the pre-trained ResNet model
        resnet = models.resnet152(pretrained=True)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(num_features,num_classes),
            nn.Softmax(dim=1)  #single label 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
#%%
#F1 score calculator 
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

def get_f1_score(net, data_loader):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            output = net(imgs)

            _, predicted = torch.max(output, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    #-average F1 score
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')

    return micro_f1

#%%


#Training function 
def train(model, train_set, valid_set, test_set, batch_size=1000, num_epochs=5, learning_rate=1e-3, decay_rate=0.9, threshold=0.5):
    torch.manual_seed(1000)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Length of train_loader",len(train_loader))
    print("Length of val_loader",len(val_loader))
    print("Length of test_loader",len(test_loader))

    # Sending data to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    i=0.0

    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0

        # Learning rate decay calculation
        current_learning_rate = learning_rate / (1 + decay_rate * epoch)

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

        for imgs, labels in train_loader:
            #labels = labels.float()
            if torch.cuda.is_available():
                imgs = imgs.cuda().to(device)
                labels = labels.cuda().to(device)

            optimizer.zero_grad()
 
            out = model(imgs)
            
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            i+=1

        total_train_loss += loss.item()

        train_epoch_acc = get_f1_score(model, train_loader)
        train_acc.append(train_epoch_acc)

        val_epoch_acc = get_f1_score(model, val_loader)
        val_acc.append(val_epoch_acc)

        train_loss.append(total_train_loss / (i + 1))
        val_loss.append(evaluate(model, val_loader, criterion))

        try:
            print(("Epoch {}: Train Acc: {:.4f}, Train Loss: {:.4f} | " +
                "Validation Acc: {:.4f}, Validation Loss: {:.4f}").format(
            epoch + 1,
            train_acc[epoch],
            train_loss[epoch],
            val_acc[epoch],
            val_loss[epoch]))
        except Exception as e:
            print("An error occurred during printing:", e)
        
        
        if(epoch % 5) == 4 or epoch == 0:
                model_path = get_model_name(model.name, batch_size, learning_rate, decay_rate, threshold, epoch)
                torch.save(model.state_dict(), model_path)

                with open("{}_train_acc.csv".format(model_path), "w", newline="") as f_train_acc:
                    writer_train_acc = csv.writer(f_train_acc)
                    writer_train_acc.writerow(train_acc)
        

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    try:
        with open("{}_train_acc.csv".format(model_path), "w", newline="") as f_train_acc:
            writer_train_acc = csv.writer(f_train_acc)
            writer_train_acc.writerow(train_acc)

        with open("{}_val_acc.csv".format(model_path), "w", newline="") as f_val_acc:
            writer_val_acc = csv.writer(f_val_acc)
            writer_val_acc.writerow(val_acc)
    except Exception as e:
        print("An error occurred while saving the results:", e)
    
    return test_loader

#%%
def get_model_name(name, batch_size, learning_rate, decay_rate, threshold, epoch):
    path = "model_{0}_bs{1}_lr{2}_dr{3}_thresh{4}_epoch{5}".format(name, batch_size, learning_rate, decay_rate, threshold, epoch)

    return path

def evaluate(model, loader, criterion):
    total_loss = 0
    i = 0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.cuda().to(device) # Move the input tensors to the desired device
            labels = labels.cuda().to(device)

            #labels = labels.float()
            out = model(imgs)  # forward pass
            loss = criterion(out, labels)  # compute the total loss
            total_loss += loss.item()
            i += 1

            
    model.train()  # Set the model back to training mode

    loss = float(total_loss) / (i + 1)
    return loss

def plot_training_curve(path):
    train_acc = np.loadtxt("{}_train_acc.csv".format(path), delimiter=',')  # Specify delimiter as ','
    val_acc = np.loadtxt("{}_val_acc.csv".format(path), delimiter=',')  # Specify delimiter as ','

    plt.title("Train vs Validation Accuracy")
    n = len(train_acc)  # number of epochs
    plt.plot(range(1, n + 1), train_acc, label="Train")
    plt.plot(range(1, n + 1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

#%%

resnetmodel = ResNet(num_classes).cuda()
print(torch.cuda.is_available())
path_to_resnet = get_model_name("resnetclassifier",batch_size=114,learning_rate=6e-4, decay_rate=0.9, threshold=0.7, epoch=19)

torch.cuda.empty_cache()
classifier_trained = train(resnetmodel,train_set, valid_set, test_set, batch_size=114, num_epochs=20, learning_rate=6e-4, decay_rate=0.9, threshold=0.7)

accuracy = get_f1_score(resnetmodel, classifier_trained)
print("The final test accuracy of our model is:", accuracy*100,"%")
plot_training_curve(path_to_resnet)
# %%
