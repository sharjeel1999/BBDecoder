import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch

class GradAnalyzer():
    def __init__(self):
        pass

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


