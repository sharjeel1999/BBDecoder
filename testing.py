import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, vit_b_16

from tqdm import tqdm

# Define the CNN architecture
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, num_classes)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
        
#         x = x.view(x.size(0), -1)
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = SimpleCNN(num_classes=10)
model = resnet50()
# model = vit_b_16()

from BBDecoder import Master_analyzer
from torch.optim import Adam

save_path = 'save_folder'
wrapped_model = Master_analyzer(model,
                                optimizer = Adam(model.parameters(), lr=0.001),
                                save_folder = save_path)

print(wrapped_model)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='O:\DATASETS',
    train=True,
    download=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='O:\DATASETS',
    train=False,
    download=True,
    transform=transform
)

subset_indices = list(range(96))  # Indices of the first 10 samples
subset = torch.utils.data.Subset(testset, subset_indices)

testloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


wrapped_model.initialize_analyser(layer_inds = [0, 1, 2, 3, 4],
                                grad_flag = True,
                                grad_hist_flag = True,
                                track_grads = True,
                                function_flag = False)

## Testing using a dummy dataset
Epochs = 5
for epoch in range(Epochs):
    
    losses = []
    for data in tqdm(testloader):
        inputs, labels = data
        wrapped_model.optimizer.zero_grad()
        outputs = wrapped_model.forward_propagation(inputs)
        loss = F.cross_entropy(outputs, labels)
        wrapped_model.backward_propagation(loss)
        
        losses.append(loss.item())
    
    print('Epoch: ', epoch)
    print('Average Loss: ', sum(losses)/len(losses))


# wrapped_model.visualize_weight_hist('O:\\PCodes\\black_box\\save_folder\\Weights')
# wrapped_model.threshold_pruning(0.01)
# wrapped_model.visualize_weight_hist('O:\\PCodes\\black_box\\save_folder\\Weights_2')

wrapped_model.save_tracked_data()

# for name, module in wrapped_model.model.named_children():
#     if module.Trainable:
#         print('name: ', name)
#         print('module: ', module)
#         print('module index: ', module.index)
#         print('module name: ', module.name)
#         print('module track flag: ', module.track_flag)
#         print('module main layer: ', module.main_layer)
#         print('module master tracker: ', module.master_tracker)
#         print('module master tracker L1: ', module.master_tracker['L1'])
#         print('module master tracker L2: ', module.master_tracker['L2'])
#         print('module master tracker L1 len: ', len(module.master_tracker['L1']))
#         print('module master tracker L2 len: ', len(module.master_tracker['L2']))