from .config import get_conf, get_int
import torch
import pandas as pd
from .dataset import Dataset
from .transformer import FeedForward, Embeddings, EncoderLayer, DecoderLayer, MultiHeadAttention, Transformer, AdamWarmup, LossWithLS, device
from .evaluate import evaluate

class Predictor :
    
    def __init__(self):
        path = get_conf("path", "training")
        epochs = get_int("training", "epochs")
        #checkpoint = torch.load(f"{path}/checkpoint_{str(epochs-1)}.pth.tar")
        #self.transformer = checkpoint['transformer']

    def predict(self, input) : 
            return 'Welcome'
            ds = Dataset()
            processor = ds.get_processor()
            input = ds.encode_source(input, processor)
            input = torch.LongTensor(input).to(device).unsqueeze(0)
            input_mask = (input!=0).to(device).unsqueeze(1).unsqueeze(1)  
            return evaluate(self.transformer, input, input_mask, int(40))