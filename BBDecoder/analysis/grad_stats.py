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
                            nn = n.replace('main_layer', module.name)
                            layers.append(nn)
                            ave_grads.append(p.grad.abs().mean().item())
                            max_grads.append(p.grad.abs().max().item())
                            if self.grad_hist_flag:
                                self.generate_hist(p, module.index)
                        except:
                            ave_grads.append(0)
                            max_grads.append(0)

        return np.array(ave_grads), np.array(max_grads), np.array(layers)
    
        # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        # plt.xlim(left=0, right=len(ave_grads))
        # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        # plt.xlabel("Layers")
        # plt.ylabel("average gradient")
        # plt.title("Gradient flow")
        # plt.grid(True)
        # plt.legend([Line2D([0], [0], color="c", lw=4),
        #             Line2D([0], [0], color="b", lw=4),
        #             Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        # plt.tight_layout()

        # save_path = os.path.join(self.save_folder, 'Grad_graph.jpg')
        # plt.savefig(save_path)


