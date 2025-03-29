
## Features to add
- include the implementaion of the following [paper](https://github.com/aaronserianni/attention-iou?utm_source=tldrai), to measure the biases in the network.
- Grad mean, max.
- Grad variance for each itteration and each epoch.
- L1 and L2 norms of gradients.
- Check dead or close to dead neurons and setup threshold based network pruning.
- weight distribution overtime (weight updates and magnitudes overtime).
- Saturation in activation functions (percentage of neuron values saturated).
- variance across channels (accross a certain selected dimension).
- Check operation being performed (function graph). Create a separate activation wrapper?

## ETC
- Mean of each itteration is not added (for gradient graphs, L1 and L2 norms)

<br>
<br>
<br>
<br>

# BBDecoder
<!-- ![Main Image](Images/unnamed.png) -->

## Initialization

Start by wrapping the entire model. Each layer in the model is assigned a name and a unique layer ID. 
```
from BBDecoder import Master_analyzer

model = resnet50().to(device)
wrapped_model = Master_analyzer(model)
```

## Gradient analysis

`wrapped_model.backward_propagation()` can be used to get gradients of each layer and also the L1 and L2 norms of the gradient for each itteration.

```
for data in tqdm(testloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    outputs = wrapped_model.forward_propagation(inputs)
    loss = F.cross_entropy(outputs, labels)

    optimizer.zero_grad()
    wrapped_model.backward_propagation(loss, collect_grads = True, layers = [0, 1, 2, 3, 4, 5, 6])
    optimizer.step()
    losses.append(loss.item())
    
print('Epoch: ', epoch)
print('Average Loss: ', sum(losses)/len(losses))
wrapped_model.save_collected_grads(save_folder, ep = epoch)
```

The graphs will be saved directly in the `save_folder` path. `collect_grads = True` is grads need to be collected other wise `False`. `layers` is the list of layers that need to be analyzed

![Grad](Images/model_gradients.jpg)

_Mean and Max gradients through layers accross the entire dataset._

L1 and L2 norms of each selected layer in the model can highlight the stability of the training process.

![Grad](Images/Gradient_norms.jpg)

_L1 and L2 norms differentiating between stable and unstable training process._

## Weight histograms

`wrapped_model.visualize_weight_hist(save_folder, layer_inds)` can be used to save weight histograms for layers with unique ID specified in `layer_inds`.

Weight histograms can be used to identify saturation, divergence and clustering in layer weights. Weights clustering at extreme ends can saturate activation functions. Clustering of weights around certain values in convolution operation can point towards kernels with similar values, this would prevent the network to distinguish between subtle features.

![Grad](Images/weight_hist.jpg)

_Weight histograms of multi-layer perceptrons and convolutional layers._
