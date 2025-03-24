import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from BBDecoder import Master_analyzer
from torch.optim import Adam

save_path = 'save_folder'
model = SimpleCNN(num_classes=10)
wrapped_model = Master_analyzer(model,
                                optimizer = Adam(model.parameters()),
                                save_folder = save_path,
                                layer_inds = [0,1,2,3,4],
                                grad_flag = True,
                                grad_hist_flag = True,
                                track_grads = True,
                                function_flag = False)


## Testing using a dummy dataset
Epochs = 10
for epoch in range(Epochs):
    print('Epoch: ', epoch)
    x = torch.randn(4, 1, 28, 28)
    y = wrapped_model.forward_propagation(x)
    loss = torch.sum(y)
    wrapped_model.backward_propagation(loss)
    print('Loss: ', loss)

wrapped_model.visualize_weight_hist('O:\\PCodes\\black_box\\save_folder\\Weights')
wrapped_model.threshold_pruning(0.01)
wrapped_model.visualize_weight_hist('O:\\PCodes\\black_box\\save_folder\\Weights_2')

# from view.torchview import draw_graph
# from torchviz import make_dot
# from visualizer import create_graphviz_graph, list_layers

# if __name__ == "__main__":
#     model = SimpleCNN(num_classes=10)
#     print(model)

#     for z, (name, module) in enumerate(model.named_children()):
#         # Check if the module is a leaf node (e.g., a layer like nn.Linear)
#         if isinstance(module, nn.Module) and not list(module.children()):  
#             # Replace the layer with its wrapped version
#             setattr(model, name, Main_wrapper(module, name, z))
#         else:
#             # Recursively wrap layers in submodules (if any)
#             Main_wrapper(module)

#     print(model)
#     # model_graph = draw_graph(model, input_size = (4, 1, 28, 28), expand_nested = True, save_graph = True, depth = 4, hide_inner_tensors=True)
#     # model_graph.visual_graph

#     print('---------------------------------')
#     list_layers(model)

#     x = torch.randn(4, 1, 28, 28)
#     # create_graphviz_graph(model, x)
    
#     outputs = model(x)
#     print("Output shape:", outputs.shape)
