from .visualizer import list_layers
from .analysis import GradAnalyzer, LayerAnalyzer
from .wrappers import Main_wrapper
import torch.nn as nn

class Master_analyzer(GradAnalyzer, LayerAnalyzer):
    def __init__(self,
                 model,
                 optimizer,
                 save_folder,
                 layer_inds = None,
                 grad_flag = False,
                 grad_hist_flag = False,
                 track_grads = False,
                 function_flag = False,
                 ):
        """
        model: Model to be analyzed.
        optimizer: Optimizer used to train the model.
        save_folder: Folder to save all results and plots.
        layer_inds: List of indices of the layers to be analyzed.
        grad_flag: Flag to plot the gradient flow.
        function_flag: Flag to plot the function flow.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.save_folder = save_folder
        self.layer_inds = layer_inds
        self.grad_flag = grad_flag
        self.grad_hist_flag = grad_hist_flag
        self.track_grads = track_grads
        self.function_flag = function_flag

        print('----- List of layer and their indices -----')
        self.wrap_layers()
        list_layers(self.model)

        self.tracker = {}

    def wrap_layers(self):
        for z, (name, module) in enumerate(self.model.named_children()):
            # Check if the module is a leaf node (e.g., a layer like nn.Linear)
            if isinstance(module, nn.Module) and not list(module.children()):  
                # Replace the layer with its wrapped version
                setattr(self.model, name, Main_wrapper(module, name, z, self.track_grads))
            else:
                # Recursively wrap layers in submodules (if any)
                Main_wrapper(module)
        
        print('----- Wrapped Layers -----')

    def forward_propagation(self, x):
        return self.model(x)
    
    def backward_propagation(self, loss):
        loss.backward()

        if self.layer_inds is not None:
            if self.grad_flag:
                self.check_grads()

        self.optimizer.step()
        self.optimizer.zero_grad()