from .visualizer import list_layers, draw_graph
from .analysis import GradAnalyzer, LayerAnalyzer
from .wrappers import Main_wrapper

import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

from ptflops import get_model_complexity_info

class Master_analyzer(nn.Module, GradAnalyzer, LayerAnalyzer):
    def __init__(self,
                 model,
                 save_path,
                 input_size,
                 depth = 3,
                 modular = True
                 ):
        """
        Wrappes the entire model and contains functions to visualize and analyze the wrapped model
        Args:
            model: Model to be analyzed.
            input_size: Input size of the model.
        """
        super(Master_analyzer, self).__init__()
        self.model = model
        self.input_size = input_size
        self.save_path = save_path
        self.modular = modular
        self.depth = depth

        self.layer_inds = None
        self.vid_out = None

        self.ave_grads = []
        self.max_grads = []
        self.l1_norm = []
        self.l2_norm = []

        self.layer_names = []

        self.cur_depth = 0
        self.wrap_layers()

        self.summary()
        # self.visualize_architecture()

        # I need to be able to access things like:
        # 1) list of all indices and names of layers

    def summary(self):
        print('----- List of layer and their indices -----')
        list_layers(self.model)

        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f'----- Total model parameters: {total_params} -----')

        macs, params = get_model_complexity_info(self.model, self.input_size, as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
        
        print(f'Computational complexity: {macs}')
        print(f'Number of parameters: {params}')



    def visualize_architecture(self):
        model_graph = draw_graph(self, input_size = self.input_size, expand_nested=True, hide_module_functions=True,
                                 directory = self.save_path, depth = 5)
        model_graph.visual_graph.render("Model_architecture", format = "png")
        print('-------- Model architecture saved -----')

    def initiate_gradcam(self, layers):
        self.model.eval()
        for name, module in self.model.named_children():
            if module.index in layers:
                module.gradcam_flag = True
                module.initiate_hooks()


    def forward(self, x, *args, **kwargs):
        # this is just for torchview visualization
        return self.model(x, *args, **kwargs)


    def wrap_layers(self):
            
            for z, (name, module) in enumerate(self.model.named_children()):
                self.layer_names.append(name)
                
                if isinstance(module, nn.Module) or isinstance(module, nn.Sequential):#and not list(module.children()):  
                    setattr(self.model, name, Main_wrapper(module, name, z))

    # def wrap_layers(self, module = None, pz = None):
    #     Work on this afterwards
    #     if module is None:
    #         module = self.model
        
    #     named_children_copy = list(module.named_children())

    #     for z, (name, module) in enumerate(named_children_copy):
    #         self.layer_names.append(name)

    #         if pz == None:
    #             cind = z
    #         else:
    #             cind = str(pz) + '.' + str(z)

    #         if self.cur_depth < self.depth:
    #             if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential) and not isinstance(module, Main_wrapper):
    #                 setattr(self.model, name, Main_wrapper(module, name, str(cind)))  
    #             else:
    #                 self.cur_depth += 1
    #                 self.wrap_layers(module, cind)
            
    #         else:
    #             setattr(self.model, name, Main_wrapper(module, name, str(cind))) ##### just self.model not enough.....


    def forward_propagation(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
    
    def backward_propagation(self, loss, collect_grads = False, layer_inds = None):
        """
        Calculates gradients and stores for specified layers.

        Args:
            loss: loss between prediction and ground truth.
            collect_grad: Beggins to collect gradients, L1 and L2 norms if collect_grad = True.
            layers: List of layers to be processed.
        """
        loss.backward()

        if layer_inds is not None:
            if collect_grads:
                self.collect_grads(layer_inds)



    # save_tracked_data() is no longer needed collect_grads() in GradAnalyzer does this.
    # CONFIRM AND DELETE LATER
    def save_tracked_data(self, save_folder):
        data = []
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
        save_path = os.path.join(save_folder, 'tracked_grads.csv')
        df.to_csv(save_path, index = False)


    # Runs a forward prop, so it must stay here
    def get_sim(self, x, layer_inds, dim, sim_method = 'kl_divergence'):
        """
        Calculates the similarity between the features across dim.

        Args:
            x: A 4D tensor of shape [batch_size, channels, height, width].
            layers: A list containing the layers that need to be processed.
            dim: Dimension to calculate the similarity across.
            sim_method (str, optional): Specifies the method to apply for similarity:
                'cosine' | 'kl_divergence'. Default: 'kl_divergence'.
        """
        self.model.eval()
        for name, module in self.model.named_children():
            if module.index in layer_inds:
                module.record_sim = True
                module.sim_method = sim_method
                module.sim_dim = dim

        with torch.no_grad():
            _ = self.model(x)

        for name, module in self.model.named_children():
            if module.index in layer_inds:
                print(f'Layer Index: {module.index}, Layer Name: {module.name}, Similarity Scores: {module.sim_scores}')
    


    def initiate_feature_recoding(self, layer, path, post_proc = None):
        """
        Initiates the recording of intermediate features for the specified layer.

        Args:
            layer: Layer to be processed.
        """
        self.model.eval()
        for name, module in self.model.named_children():
            if module.index == layer:
                module.record_inter_features = True
                module.inter_features_path = path
                if post_proc is not None:
                    module.post_proc_function = post_proc
                    
                # if dim is not None:
                #     module.record_dim = dim
                break

    # Maybe do this for a batch of inputs instead of a single test_input
    def get_inter_features(self, test_input, layer, path, post_proc = None):
        """
        Returns the intermediate features of the specified layer.

        Args:
            test_input: Input tensor to be passed through the model.
            layer: Layer to be processed.
            path: Folder Path to save the intermediate features.
        """

        self.initiate_feature_recoding(layer, path, post_proc)

        with torch.no_grad():
            _ = self.model(test_input)


    # initiate recording before calling this function. initiate recoding -> train -> compile_feature_evolution
    def compile_feature_evolution(self, layer, fps = 5):
        """
        Compiles the intermediate features of the specified layer.

        Args:
            test_input: Input tensor to be passed through the model.
            layer: Layer to be processed.
            path: Folder Path to save the intermediate features.
            channel: Channel to be processed. If None, randomly selects a channel.
        """
        
        for name, module in self.model.named_children():
            if module.index == layer:
                feats_archive = module.feats_archive
                flows = feats_archive.get_all_flows()

                width, height = module.get_frame_size()
                vid_out = cv2.VideoWriter(self.save_path + f'/Layer{module.index}_feature_evolution.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
                
                if not vid_out.isOpened():
                    print('Error: Could not open video writer')
                    

                for sample in flows:
                    image = cv2.imread(sample)

                    if image is None:
                        print(f'Could not opend image at path: {sample}')
                    
                    if image is not None:
                        vid_out.write(image)
        
                if vid_out is not None:
                    vid_out.release()
    
    def GradCAM_run(self, layer, input_tensor, target_class = None):
        """
        Generates a Grad-CAM heatmap for the specified layer.

        Supports classification outputs shaped (N, C) and segmentation outputs
        shaped (N, C, H, W) by reducing the selected class scores to a scalar
        before backpropagation.
        """

        self.model.eval()
        target_module = None
        for name, module in self.model.named_children():
            if module.index == layer:
                module.gradcam_flag = True
                module.initiate_hooks()
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer with index {layer} not found.")

        output = self.model(input_tensor)

        if output.dim() == 4:  # Segmentation: (N, C, H, W)
            if target_class is None:
                target_class = torch.sum(output, dim=(2, 3)).argmax(dim=1)[0].item()
            target_score = output[:, target_class, :, :].mean()

        elif output.dim() == 2:  # Classification: (N, C)
            if target_class is None:
                target_class = output.argmax(dim=1)[0].item()
            target_score = output[:, target_class].mean()

        else:
            raise ValueError(f"Unsupported output shape {tuple(output.shape)} for Grad-CAM.")

        self.model.zero_grad()
        target_score.backward()

        heatmap = target_module.grad_archive.get_local_heatmap()
        return heatmap
                        
        