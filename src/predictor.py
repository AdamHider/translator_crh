import torch
import re
from evaluate import evaluate
from transformer import *
from dataset import Dataset
from constants import *
#import nltk
from nltk import tokenize
#nltk.download('punkt')

class Predictor :
    def __init__(self) :
        checkpoint = torch.load(f'{CKPT_DIR}/checkpoint_big_{num_epochs - 1}.pth.tar')
        self.transformer = checkpoint['transformer']

    def predict(self, input) :
        sentences = self.sentence_split(input)
        predictions = []
        for sentence in sentences :
            prediction = self.get_result(sentence)
            predictions.append(prediction)
        result = " ".join(predictions)
        return result


    def get_result(self, input): 
        ds = Dataset()
        processor = ds.get_processor()
        input = ds.encode_source(input, processor)
        input = torch.LongTensor(input).to(device).unsqueeze(0)
        input_mask = (input!=0).to(device).unsqueeze(1).unsqueeze(1)  
        prediction = evaluate(self.transformer, input, input_mask, int(40), processor)
        return self.process_prediction(prediction)

    def process_prediction(self, prediction) :
        result = re.sub(r' ([!.?,])', r'\1', prediction)
        return result

    def sentence_split(self, text) :
        return tokenize.sent_tokenize(text)
