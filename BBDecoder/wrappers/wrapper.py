import torch
import torch.nn as nn

import os
import matplotlib.pyplot as plt

class Main_wrapper(nn.Module):
    def __init__(self, layer: nn.Module, name, index):
        super().__init__()

        self.index = index
        self.name = name

        self.main_layer = layer

    def forward(self, x):
        return self.main_layer(x)
    
    def gradient_histogram(self):
        """Plots a histogram of gradients for a specific layer."""

        if self.grad is not None:
            gradients = self.grad.cpu().numpy().flatten()  # Flatten the gradient tensor
            plt.figure()
            plt.hist(gradients, bins=50)  # Adjust bins as needed
            plt.title(f"Gradient Histogram - {self.name}")# - Step {step}")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            # save_path = os.path.join(save_folder, f"grad_hist_{layer_name}_step_{step}.png")
            # plt.savefig(save_path)
            # plt.close() #Close the plot to prevent overlapping.
        else:
            print(f"No gradients available for layer: {self.name}")
    