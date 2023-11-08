import torch
from .dataset import Dataset
from torch.utils.data import IterableDataset
import csv

class DatasetIterator(IterableDataset):

    def __init__(self, filename, length):
        self.filename = filename
        self.len = length


    def __len__(self):
        return self.len

    def preprocess(self, text, text2):
        ds = Dataset()
        processor = ds.get_processor()
        text_pp = torch.LongTensor(ds.encode_source(text, processor))
        text_pp2 = torch.LongTensor(ds.encode_target(text2, processor))
        
        return text_pp, text_pp2

    def line_mapper(self, line):
        return self.preprocess(line[1], line[2])
       
    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename, encoding = "utf-8")
        reader = csv.reader(file_itr)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, reader)
        
        return mapped_itr
     

