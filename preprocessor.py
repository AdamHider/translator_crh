import torch
from dataset import encode_english, encode_french, get_sp
from torch.utils.data import IterableDataset
import csv

#=============================================
#BUILDING PREPROCCESSOR

class CustomIterableDatasetv1(IterableDataset):

    def __init__(self, filename, length):

        #Store the filename in object's memory
        self.filename = filename
        self.len = length


    def __len__(self):
        return self.len

    def preprocess(self, text, text2):

        sp = get_sp()
        text_pp = torch.LongTensor(encode_english(text, sp))
        text_pp2 = torch.LongTensor(encode_french(text2, sp))
        
        return text_pp, text_pp2

    def line_mapper(self, line):
        
        text, text2 = self.preprocess(line[1], line[2])
        
        return text, text2
       
    def __iter__(self):

        #Create an iterator
        file_itr = open(self.filename, encoding="utf-8")
        reader = csv.reader(file_itr)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, reader)
        
        return mapped_itr
     

