from .visualizer import list_layers
from .analysis import GradAnalyzer, LayerAnalyzer
from .wrappers import Main_wrapper

import torch
import torch.nn as nn
import pandas as pd
import os

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
        grad_hist_flag: Flag to plot the gradient histograms.
        track_grads: Flag to track the gradients L1 and L2 norms.
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

        self.layer_names = []

        print('----- List of layer and their indices -----')
        self.wrap_layers()
        print('--------------------------------------------')
        print(self.model)
        list_layers(self.model)


    def wrap_layers(self):
        for z, (name, module) in enumerate(self.model.named_children()):
            self.layer_names.append(name)
            
            print('****** ', module)
            if isinstance(module, nn.Module) or isinstance(module, nn.Sequential):#and not list(module.children()):  
                setattr(self.model, name, Main_wrapper(module, name, z, self.track_grads))
            # else:
                # Recursively wrap layers in submodules (if any)
                # Main_wrapper(module, name, z, False)
        
        print('----- Wrapped Layers -----')


    def forward_propagation(self, x):
        return self.model(x)
    
    def backward_propagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        if self.layer_inds is not None:
            if self.grad_flag:
                self.check_grads()

        self.optimizer.step()

    def save_tracked_data(self):
        data = []

        Column_names = ['L1', 'L2']

        for name, module in self.model.named_children():
            if module.track_flag and module.Trainable:
                tracker = module.master_tracker
                # Improve the data saving process
                for i in range(len(tracker['L1'])):
                    data.append({
                        'Layer': module.name,
                        'L1': tracker['L1'][i],
                        'L2': tracker['L2'][i]
                    })

        df = pd.DataFrame(data)
        save_path = os.path.join(self.save_folder, 'tracked_grads.csv')
        df.to_csv(save_path, index = False)

    def record_sim(self, x, layers, dim, sim_method = 'cosine', ):
        for name, module in self.model.named_children():
            if module.index in layers:
                module.record_sim = True
                module.sim_method = sim_method
                module.sim_dim = dim

        with torch.no_grad():
            _ = self.model(x)
            
        