from torchvision.models import resnet50, vit_b_16

from view.torchview import draw_graph

model = vit_b_16()
print(model)
model_graph = draw_graph(model, input_size = (2, 3, 224, 224), expand_nested = True, save_graph = True)
model_graph.visual_graph
print('done')