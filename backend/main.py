import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters ####### änderbar
input_size = 784
hidden_size = 700
hidden_2_size = 650
hidden_3_size = 600
hidden_4_size = 550
hidden_5_size = 500
hidden_6_size = 450
hidden_7_size = 400
hidden_8_size = 350
hidden_9_size = 300
hidden_10_size = 250
num_classes = 10
# Wie oft training er macht
num_epochs = 2
# Speichergröße
batch_size = 100
# Learning rate
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with 11 hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_2_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_3_size, hidden_4_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_4_size, hidden_5_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_5_size, hidden_6_size)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(hidden_6_size, hidden_7_size)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(hidden_7_size, hidden_8_size)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(hidden_8_size, hidden_9_size)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(hidden_9_size, hidden_10_size)
        self.relu10 = nn.ReLU()
        self.fc11 = nn.Linear(hidden_10_size, num_classes)
        self.relu11 = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.fc7(out)
        out = self.relu7(out)
        out = self.fc8(out)
        out = self.relu8(out)
        out = self.fc9(out)
        out = self.relu9(out)
        out = self.fc10(out)
        out = self.relu10(out)
        out = self.fc11(out)
        out = self.relu11(out)
        return out


model = NeuralNet(input_size, hidden_size, hidden_2_size, num_classes).to(device)
# Save the model checkpoint
torch.save(model.state_dict(), 'netz.pt')
model.load_state_dict(torch.load('netz.pt'))
model.eval()




# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                   

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'netz.pt')



