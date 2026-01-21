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

class Master_analyzer(nn.Module, GradAnalyzer, LayerAnalyzer):
    def __init__(self,
                 model,
                 save_path,
                 input_size = (1, 3, 32, 32),
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
        # print('----- List of layer and their indices -----')
        # list_layers(self.model)
        self.visualize_architecture()

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


    def collect_grads(self, layer_inds):
        self.layer_inds = layer_inds
        
        iter_ave, iter_max, l1_norm, l2_norm, rec_layers = self.check_grads()
        self.rec_layers = rec_layers
        if len(self.ave_grads) == 0:
            self.ave_grads = iter_ave
            self.max_grads = iter_max
            self.l1_norm = l1_norm
            self.l2_norm = l2_norm
        else:
            self.ave_grads = np.vstack((self.ave_grads, iter_ave))
            self.max_grads = np.vstack((self.max_grads, iter_max))
            self.l1_norm = np.vstack((self.l1_norm, l1_norm))
            self.l2_norm = np.vstack((self.l2_norm, l2_norm))


    def save_collected_grads(self, ep, save_folder = None):
        max_grads = np.max(self.max_grads, axis = 0)
        ave_grads = np.mean(self.ave_grads, axis = 0)

        plt.figure(figsize=(15, 6)) # (width, height)

        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), self.rec_layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        if save_folder == None:
            save_folder = os.path.join(self.save_path, 'Gradient_saves')
            
        save_path = os.path.join(save_folder, f'Epoch_{ep}_Grad_graph.jpg')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_path)
        plt.close()

        self.plot_paired_lines_from_arrays(save_folder, ep)

        self.ave_grads = []
        self.max_grads = []
        self.rec_layers = 0
        self.l1_norm = []
        self.l2_norm = []


    def plot_paired_lines_from_arrays(self, save_dir, ep):
        
        norm_save_path = os.path.join(save_dir, 'norms')
        if not os.path.exists(norm_save_path):
            os.makedirs(norm_save_path)

        a, b = self.l1_norm.shape  # Get the dimensions (both arrays have the same shape)

        for i in range(b):
            plt.figure(figsize=(10, 6))  # Create a new figure for each pair of lines
            plt.title(f"Gradient norm (Layer {self.rec_layers[i]})")  # Unique title for each subplot
            plt.xlabel('Itterations')
            plt.ylabel("Gradient Norm")

            # Plot the two lines on the same graph
            plt.plot(self.l1_norm[:, i], label=f'L1', color='blue')
            plt.plot(self.l2_norm[:, i], label=f'L2', color='orange')

            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(norm_save_path, f'Epoch_{ep}_{self.rec_layers[i]}.jpg')
            plt.savefig(save_path)
            plt.close()

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
                        
        