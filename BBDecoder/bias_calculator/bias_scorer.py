import torch
import torchvision.transforms.v2 as transforms

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
    def __init__(self, model, test_loader, bias_calculator, device, resize_attentions):
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
        self.device = device
        self.resize_attentions = resize_attentions

    def get_maps(self, image, targets, target_layers):
        cam = self.cams[self.bias_calculator](model = self.model, target_layers = target_layers)

        attention_map = torch.tensor(cam(input_tensor = image, targets = targets), device=self.device)

        if self.resize_attentions:
            attentions = transforms.functional.resize(
                torch.squeeze(attentions),
                size = image.shape[-2:],
                interpolation = transforms.InterpolationMode.BILINEAR,
            )
        else:
            attentions = torch.squeeze(attentions)
        
        attentions -= attentions.min(1)[0].min(1)[0].unsqueeze(1).unsqueeze(1)
        attentions /= 1e-7 + attentions.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1)
        attentions = torch.clamp(attentions, min=0, max=1)
        
        return attentions