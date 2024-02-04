import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import time
from torch.utils.data import DataLoader


# ------------------------------------------------ Data loading process ------------------------------------------------ //
transform = transforms.Compose([ 
    #Define image transformations for data preprocessing
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Define batch size for data loading
batch_size = 64
# Load the training dataset and create a data loader

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=256, shuffle=True, pin_memory=True)
# Load the testing dataset and create a data loader
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=True)



# Define the neural network encoder architecture
class encoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = encoder.encoder
        self.fc1 = nn.Linear(512*4*4, 4096)  # output = 1x392
        self.fc2 = nn.Linear(4096, 4096)  # output = 1x784
        self.fc3 = nn.Linear(4096, 100)  # output = 1x784


    def decode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

    def encode(self, x):
        x= self.encoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1) 
        x = self.decode(x)
        return x





if __name__ == '__main__':
 # Check if CUDA (GPU) is available and print a message
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("nah")
        # Set the device (CPU or GPU) for model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the model (Net) and load the encoder's pre-trained weights
    model = Net().to(device)
    model.encoder.load_state_dict(torch.load('encoder.pth'))

    
    # Loss Function
    criterion = nn.CrossEntropyLoss()   # Define the loss function (Cross-Entropy) for training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9)# Set up the optimizer (Stochastic Gradient Descent) with learning rate and momentum
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min') 
    num_epochs = 64 # Number of training epochs
    
    def train():
        print("Training yeeehaw!")
        model.train()# Set the model in training mode
        losses_train = []# List to store training losses
        start_time = time.time()  # Start the timer
        
        for epoch in range(1,num_epochs+1):  # loop over the dataset multiple times
            print("Epoch", epoch)
            loss_train = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()# Zero the gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels) # Compute the loss
                loss.backward()# Backpropagation
                optimizer.step()# Update model weights
                loss_train += loss.item()# Accumulate the loss

            scheduler.step(loss_train)# Adjust learning rate based on loss reduction
            losses_train.append(loss_train/len(train_loader)) # Store the average loss

            print(f"Epoch {epoch + 1}, Loss: {loss_train / len(train_loader)}")

        end_time = time.time()  # End the timer
        total_time = end_time - start_time
        print(f"Training time: {total_time/60.0:.2f} minutes.")
        
        plt.plot(range(1,num_epochs + 1), losses_train, label="train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Training Loss')
        plt.legend(loc=1)
        plt.savefig("Loss_Plot") # loss.MLP.8.png
        plt.show()



    model = Net().to(device)
    model.load_state_dict(torch.load('cifar_net.pth'))
    model.eval()
    def evaluate(model, test_loader):
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

       

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
                _, predicted_top5 = outputs.topk(5, 1)
                for i in range(labels.size(0)):
                    if labels[i] in predicted_top5[i]:
                        correct_top5 += 1

        top1_error = 1 - correct_top1 / total
        top5_error = 1 - correct_top5 / total
        top1_accuracy = correct_top1 / total
        top5_accuracy = correct_top5 / total

   # Print and display evaluation results
        print(f"Top-1 Error: {top1_error * 100:.2f}%")
        print(f"Top-5 Error: {top5_error * 100:.2f}%")
        print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
# Call the evaluate function to assess the model's performance on the test dataset
    evaluate(model, test_loader)