import torch

class GradArchive():
    """An archive that stores gradients of solutions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = None
        self.gradients = None

    def _hook_activations(self, module, input, output):
        self.activations = output#.detach()
    
    def _hook_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]#.detach()

    def get_local_heatmap(self):
        if self.activations is None or self.gradients is None:
            raise ValueError("Activations or gradients have not been recorded. Ensure that you initiate gradcam.")
        
        # Compute the weights
        weights = torch.mean(self.gradients, dim = (2, 3), keepdim=True)  # Global average pooling over width and height
        
        # Compute the weighted sum of activations
        weighted_activations = weights * self.activations
        heatmap = torch.sum(weighted_activations, dim = 1)  # Sum over the channel dimension
        
        # Apply ReLU
        heatmap = torch.relu(heatmap)
        
        # Normalize the heatmap to [0, 1]
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max - heatmap_min > 0:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = torch.zeros_like(heatmap)
        
        return heatmap.cpu().detach().numpy()