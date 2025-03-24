import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        max_grads= []
        layers = []
        for name, module in self.model.named_children():
            
            if module.index in self.layer_inds:
                for n, p in module.named_parameters():
                    if(p.requires_grad) and ("bias" not in n):
                        try:
                            layers.append(n)
                            ave_grads.append(p.grad.abs().mean().item())
                            max_grads.append(p.grad.abs().max().item())
                            if self.grad_hist_flag:
                                self.generate_hist(p, module.index)
                        except:
                            ave_grads.append(0)
                            max_grads.append(0)

        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
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

        save_path = os.path.join(self.save_folder, 'Grad_graph.jpg')
        plt.savefig(save_path)






def plot_grad_flow(named_parameters, layer_inds, folder_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    loss.backward()
    plot_grad_flow(model.named_parameters())
    optimizer.step()
    
    '''
    # print('---- entered ----')
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        print('named parameters: ', n)
        # print('parameters: ', p)
        print('parameters name/index: ', p.name, p.index)
        if(p.requires_grad) and ("bias" not in n):
            try:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
            except:
                ave_grads.append(0)
                max_grads.append(0)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    save_path = os.path.join(folder_path, 'Grad_graph.png')
    plt.savefig(save_path)