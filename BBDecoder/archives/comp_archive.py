from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

class CompArchive():
    def __init__(self):
        self.memory = None
        self.flops = None
        self.inf_time = None

    def record_comps(self, model, x):
        self.flops = FlopCountAnalysis(model, x).total()