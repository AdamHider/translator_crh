from .config import get_conf, get_int
import torch
import pandas as pd
from .dataset import Dataset
from .transformer import FeedForward, Embeddings, EncoderLayer, DecoderLayer, MultiHeadAttention, Transformer, AdamWarmup, LossWithLS, device
from .evaluate import evaluate

class Predictor :
    
    def __init__(self):
        path = get_conf("path", "training")

    def predict(self, input) : 
        path = get_conf("path", "training")
        epochs = get_int("training", "epochs")
        checkpoint = torch.load(f"{path}/checkpoint_{str(epochs-1)}.pth.tar")
        transformer = checkpoint['transformer']

        ds = Dataset()
        processor = ds.get_processor()
        english = ds.encode_source(input, processor)
        english = torch.LongTensor(english).to(device).unsqueeze(0)
        english_mask = (english!=0).to(device).unsqueeze(1).unsqueeze(1)  
        return evaluate(transformer, english, english_mask, int(40))