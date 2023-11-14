

from torch.utils.data import IterableDataset
from dataset import Dataset
import torch
import csv

class DatasetIterator(IterableDataset):

    def __init__(self, filename, length, processor):
        self.filename = filename
        self.len = length
        self.processor = processor


    def __len__(self):
        return self.len
 
    def preprocess(self, text, text2):
        ds = Dataset()
        text_pp = torch.LongTensor(ds.encode_source(text, self.processor))
        text_pp2 = torch.LongTensor(ds.encode_target(text2, self.processor))
        
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
     