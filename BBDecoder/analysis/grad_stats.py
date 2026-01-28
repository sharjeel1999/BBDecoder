import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch

class GradAnalyzer():
    def __init__(self):
        self.ave_grads = []
        self.max_grads = []
        self.l1_norm = []
        self.l2_norm = []
        self.layer_inds = []

    def collect_grads(self, layer_inds):
        self.layer_inds = layer_inds
        # OR maybe instead of using check_grads() function perform this for each layer using
        # grad_archive stored in each layer wrapper. CHECK WHICH IS MORE EFFICIENT!!!!!!!
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
        # Max of the Max grads !! or mean of max grads??
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
        self.rec_layers = None
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
            plt.xlabel('Iterations')
            plt.ylabel("Gradient Norm")

            # Plot the two lines on the same graph
            plt.plot(self.l1_norm[:, i], label=f'L1', color='blue')
            plt.plot(self.l2_norm[:, i], label=f'L2', color='orange')

            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(norm_save_path, f'Epoch_{ep}_{self.rec_layers[i]}.jpg')
            plt.savefig(save_path)
            plt.close()


    def generate_hist(self, p, index):
        if p.grad is not None:
            gradients = p.grad.cpu().numpy().flatten()
            plt.figure()
            plt.hist(gradients, bins = 50)
            plt.title(f"Gradient Histogram - {index}")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(True)

            save_path = os.path.join(self.save_folder, f"histograms/grad_hist_{index}.jpg")
            plt.savefig(save_path)
            plt.close()  # Close the plot to prevent overlapping
        else:
            print(f"No gradients available for layer: {index}")

    def check_grads(self):
        ave_grads = []
        max_grads = []
        l1_grads = []
        l2_grads = []
        layers = []
        for name, module in self.model.named_children():

            if module.index in self.layer_inds:
                for n, p in module.named_parameters():
                    if(p.requires_grad) and ("bias" not in n):
                        try:
                            nn = n.replace('main_layer', module.name)
                            layers.append(nn)
                            ave_grads.append(p.grad.abs().mean().item())
                            max_grads.append(p.grad.abs().max().item())
                            l1_grads.append(p.grad.abs().sum().item())
                            l2_grads.append(torch.sqrt((p.grad**2).sum()).item())
                            # if self.grad_hist_flag:
                            #     self.generate_hist(p, module.index)
                        except:
                            ave_grads.append(0)
                            max_grads.append(0)
                            l1_grads.append(0)
                            l2_grads.append(0)

        return np.array(ave_grads), np.array(max_grads), np.array(l1_grads), np.array(l2_grads), np.array(layers)


