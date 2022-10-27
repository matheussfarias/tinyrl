#from transformers import AutoModel
import torch

def quantize(x,q):
    low = torch.min(x)
    x_shifted = (x-low)
    high = torch.max(x_shifted)
    x_shifted_scaled = x_shifted*(2**q-1)/high
    x_quantized = (torch.floor(x_shifted_scaled.detach().clone()+.5)).type(torch.int16)
    return x_quantized, (low, high)

def dequantize(x, extra_args, q):
    low, high = extra_args
    x_shifted = x.type(torch.float32)*high/(2**q-1)
    x = x_shifted + low
    return x



model = torch.load('/Users/nec368/harvard/cs249r/project/tinyrl/TinyBERT/bert-tiny-finetuned-sst2/pytorch_model.bin', map_location=torch.device('cpu'))
quantized_model = model.copy()
#model.eval()
print(model.keys())
for layer in model.keys():
    weights, extra_args = quantize(model[layer], 6)
    weights = dequantize(weights, extra_args, 6)
    quantized_model[layer] = weights

torch.save(quantized_model, 'quantized_model/pytorch_model.bin')


#for param in model.parameters():
#    print(param)