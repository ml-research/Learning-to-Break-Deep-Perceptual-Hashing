import sys
sys.path.insert(0,'/code')

import onnx
import torch
from models.neuralhash import NeuralHash

onnx_model = onnx.load('models/model.onnx')
pytorch_model = NeuralHash(onnx_model)
torch.save(pytorch_model.state_dict(), './models/model.pth')
