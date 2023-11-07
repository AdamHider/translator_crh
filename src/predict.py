import torch
import pandas as pd
from .dataset import encode_source, get_sp
from .transformer import FeedForward, Embeddings, EncoderLayer, DecoderLayer, MultiHeadAttention, Transformer, AdamWarmup, LossWithLS, device
from .evaluate import evaluate



def get_transformer() : 
    checkpoint = torch.load('checkpoint_big_7.pth.tar')
    transformer = checkpoint['transformer']
    return transformer

def predict() : 
    transformer = get_transformer()
    while(1):
        english = input("English: ") 
        if english == 'quit':
            break
        sp = get_sp()
        english = encode_source(english, sp)
        english = torch.LongTensor(english).to(device).unsqueeze(0)
        english_mask = (english!=0).to(device).unsqueeze(1).unsqueeze(1)  
        sentence = evaluate(transformer, english, english_mask, int(40))
        print(sentence+'\n')