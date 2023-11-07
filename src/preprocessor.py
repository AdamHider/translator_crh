import torch
from .dataset import encode_source, encode_target, get_sp
from torch.utils.data import IterableDataset
import csv

class CustomIterableDatasetv1(IterableDataset):

    def __init__(self, filename, length):
        self.filename = filename
        self.len = length


    def __len__(self):
        return self.len

    def preprocess(self, text, text2):
        sp = get_sp()
        text_pp = torch.LongTensor(encode_source(text, sp))
        text_pp2 = torch.LongTensor(encode_target(text2, sp))
        
        return text_pp, text_pp2

    def line_mapper(self, line):
        return self.preprocess(line[1], line[2])
       
    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename, encoding="utf-8")
        reader = csv.reader(file_itr)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, reader)
        
        return mapped_itr
     

