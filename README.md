
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


# BBDecoder
<!-- ![Main Image](Images\unnamed.png) -->

## Initialization

Start by wrapping the entire model.
```
from BBDecoder import Master_analyzer

model = resnet50().to(device)
wrapped_model = Master_analyzer(model)
```

## Gradient analysis

`wrapped_model.backward_propagation()` can be used to get gradients of each layer and also the L1 and L2 norms of the gradient for each itteration.

```
outputs = wrapped_model.forward_propagation(inputs)
loss = F.cross_entropy(outputs, labels)

optimizer.zero_grad()
wrapped_model.backward_propagation(loss, collect_grads = True, layers = [0, 1, 2, 3, 4, 5, 6])
optimizer.step()
losses.append(loss.item())
```
