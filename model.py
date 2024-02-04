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
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop images with padding
    transforms.ToTensor(), # Convert the image data to PyTorch tensors for processing.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
# Normalize the pixel values to have a mean and standard deviation of (0.5, 0.5, 0.5).
# Normalization aids in faster convergence during training.
])

batch_size = 64
# Set the batch size for training

# Create a training dataset using CIFAR-100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

# Create a data loader for the training dataset with the specified batch size, shuffling the data,
# and utilizing pin memory for faster data transfer if available.
train_loader = DataLoader(trainset, batch_size=256, shuffle=True, pin_memory=True)
# Create a testing dataset using CIFAR-100, specifying the root directory,
# enabling downloading if the dataset is not found, and applying the defined transformation.
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, pin_memory=True)

# Create a data loader for the testing dataset with the specified batch size, not shuffling the data,
# and utilizing pin memory for faster data transfer if available.




class encoder:
      # Initialize the encoder as a sequence of neural network layers.
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),# Apply a 3x3 convolution with 3 input channels
        nn.ReflectionPad2d((1, 1, 1, 1)),# Apply reflection padding to the input
        nn.Conv2d(3, 64, (3, 3)), # Apply a 3x3 convolution with 64 output channels
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),# Apply reflection padding
        nn.Conv2d(64, 64, (3, 3)),# Apply a 3x3 convolution with 64 output channels (relu1-2)
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # Perform max pooling
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = encoder.encoder
        self.fc1 = nn.Linear(512*4*4, 4096)  # output = 1x392
        self.fc2 = nn.Linear(4096, 4096)  # output = 1x784
        self.fc3 = nn.Linear(4096, 100)  # output = 1x784


    def decode(self, x):
          # Decode method: Transforms the encoded features back into the model's output.
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

    def encode(self, x):
         # Encode method: Extracts features from input data using the 'encoder' module
        x= self.encoder(x)
        return x
    
    def forward(self, x):
          # Forward method: Defines the forward pass of the model, which combines encoding and decoding.
        x = self.encode(x)# Encode the input data using the 'encoder'
        x = x.view(x.size(0), -1) # Flatten the encoded features
        x = self.decode(x) # Decode the features back into the model's output
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
    # Define the loss function (Cross-Entropy) for training
    criterion = nn.CrossEntropyLoss() 
    # Initialize a learning rate scheduler that adjusts learning rate based on loss reduction
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9)
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
                
                inputs = inputs.to(device)# Move input data to the selected device
                labels = labels.to(device)# Forward pass
                
                optimizer.zero_grad()# Zero the gradients
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()# Backpropagation
                optimizer.step() # Update model weights
                loss_train += loss.item() # Accumulate the loss

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
        plt.savefig("Loss_Plot") 
        # Save the loss plot
        plt.show()


    train() # Start the training process
    PATH = './cifar_net.pth' # Define the path to save the trained model
    torch.save(model.state_dict(), PATH)# Save the model's state dictionary