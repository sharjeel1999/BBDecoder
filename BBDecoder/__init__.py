from visualizer import list_layers
from analysis import plot_grad_flow

class Master_analyzer():
    def __init__(self, 
                 model, 
                 optimizer,
                 layer_inds = None,
                 grad_flag = False,
                 function_flag = False,
                 ):
        """
        model: Model to be analyzed.
        optimizer: Optimizer used to train the model.
        layer_inds: List of indices of the layers to be analyzed.
        grad_flag: Flag to plot the gradient flow.
        function_flag: Flag to plot the function flow.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.layer_inds = layer_inds
        self.grad_flag = grad_flag
        self.function_flag = function_flag


        print('----- List of layer and their indices -----')
        list_layers(self.model)

    def forward_propagation(self, x):
        return self.model(x)
    
    def backward_propagation(self, loss):
        loss.backward()
        if self.layer_inds is not None:
            if self.grad_flag:
                plot_grad_flow(self.model.named_parameters(), self.layer_inds)
        self.optimizer.step()
        self.optimizer.zero_grad()