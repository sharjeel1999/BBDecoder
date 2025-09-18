import torch

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
)
class BiasScorer:
    def __init__(self, model, test_loader, bias_calculator, device):
        self.cams = {
            "GradCAM": GradCAM,
            "HiResCAM": HiResCAM,
            "ScoreCAM": ScoreCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "AblationCAM": AblationCAM,
            "XGradCAM": XGradCAM,
            "EigenCAM": EigenCAM,
            "EigenGradCAM": EigenGradCAM,
            "LayerCAM": LayerCAM,
        }
        self.model = model
        self.bias_calculator = bias_calculator
        self.bias_calculator

    def get_maps(self, image, targets, target_layers):
        cam = self.cams[self.bias_calculator.cam_type](model = self.model, target_layers = target_layers)

        attention_map = torch.tensor(cam(input_tensor = image, targets = targets), device=self.device)
        