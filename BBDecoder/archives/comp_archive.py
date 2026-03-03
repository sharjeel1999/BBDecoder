from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time
import torch

class CompArchive():
    def __init__(self):
        self.memory = None
        self.flops = None
        self.inf_time = None

    def record_comps(self, model, x):
        self.flops = FlopCountAnalysis(model, x).total()

    def record_inf_time(self, layer, x, args=None, kwargs=None):
        start_time = time.time()
        with torch.no_grad():
            layer(x, *args, **kwargs)
        end_time = time.time()
        self.inf_time = end_time - start_time

    def get_comps(self):
        comps_info = {
            'flops': self.flops,
            'inf_time': self.inf_time
        }
        return comps_info
