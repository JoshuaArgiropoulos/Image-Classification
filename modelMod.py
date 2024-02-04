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
# Define data preprocessing transformations
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding
    transforms.ToTensor(),# Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# Normalize values
])
# Batch size for training
batch_size = 64
# Load the CIFAR-100 dataset with defined transformations
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=256, shuffle=True, pin_memory=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=True)



# Define an encoder using a sequence of convolutional layers
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
# Define a neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = encoder.encoder
        self.fc1 = nn.Linear(512*4*4, 4096)  # Fully connected layer
        self.fc2 = nn.Linear(4096, 4096)  
        self.fc3 = nn.Linear(4096, 100)  

# Decoder function to reconstruct the input data
    def decode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x) # Apply sigmoid activation for output
        return x
   # Encoder function to encode the input data
    def encode(self, x):
        x= self.encoder(x)
        return x
     # Forward pass through the network
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.decode(x)
        return x





if __name__ == '__main__':
 # Check if CUDA is available and set the device
    if torch.cuda.is_available():
        print("Using CUDA")
    else:
        print("nah")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create an instance of the neural network model and load the encoder's state
    model = Net().to(device)
    model.encoder.load_state_dict(torch.load('encoder.pth'))

    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9) # Define the optimizer for training the model
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min') # Define a learning rate scheduler based on loss reduction
    num_epochs = 64 # Number of training epochs
    
    def train():# Function to train the model
        print("Training yeeehaw!")
        model.train()
        losses_train = []# List to store training losses
        start_time = time.time()  # Start the timer
        
        for epoch in range(1,num_epochs+1):  # loop over the dataset multiple times
            print("Epoch", epoch)
            loss_train = 0.0# Initialize the training loss for this epoch

            for i, data in enumerate(train_loader):
                inputs, labels = data # Get input data and labels
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad() # Clear the gradients
                outputs = model(inputs)# Forward pass
                loss = criterion(outputs, labels)# Compute the loss
                loss.backward()# Backpropagation
                optimizer.step()# Update model weights

                loss_train += loss.item()# Accumulate the loss

            scheduler.step(loss_train) # Adjust the learning rate based on loss reductio
            losses_train.append(loss_train/len(train_loader)) # Store the average loss for the epoch

            print(f"Epoch {epoch + 1}, Loss: {loss_train / len(train_loader)}")

        end_time = time.time()  # End the timer
        total_time = end_time - start_time # Calculate the total training time
        print(f"Training time: {total_time/60.0:.2f} minutes.")# Print the training time in minutes
           # Plot the training loss over epochs
        plt.plot(range(1,num_epochs + 1), losses_train, label="train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Training Loss')
        plt.legend(loc=1)
        plt.savefig("Loss_Plot") # Save the loss plot to a file
        plt.show()# Display the loss plot


    train() # Call the training function
    PATH = './cifar_net.pth'   # Define a path to save the trained model's state
    torch.save(model.state_dict(), PATH) # Save the model's state dictionary to the specified path