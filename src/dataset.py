from .config import get_conf, get_int
import os
import pandas as pd
import random
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

class Dataset : 
    def __init__(self):
        path = get_conf("path", "dataset")
        self.model = f'{path}/{get_conf("dataset", "model_name")}'
        self.input_file = f'{path}/{get_conf("dataset", "input_filename")}'
        self.temp_file = f'{path}/{get_conf("dataset", "temp_filename")}'
        self.build_file = f'{path}/{get_conf("dataset", "final_filename")}'
        self.max_len = get_int("dataset", "max_len")

    def preload(self):
        random.seed(42)
        df = pd.read_csv(self.input_file, header = None, chunksize = 500000)
        df = next(df)
        df[0] = df[0].astype('str')
        df[1] = df[1].astype('str')
        tmp_dataset = list(df[0] + df[1])
        with open(self.temp_file, "w", encoding = "utf-8") as output:
            output.write(str(("\n").join(tmp_dataset)))
        return df

    def train(self):
        sentencepiece_params = ' '.join([
            '--input={}'.format(self.temp_file),
            '--model_type={}'.format("bpe"),
            '--model_prefix={}'.format(self.model),
            '--vocab_size={}'.format(10000),
            '--pad_id={}'.format(0),
            '--unk_id={}'.format(1),
            '--bos_id={}'.format(2),
            '--eos_id={}'.format(3)
        ])
        print(sentencepiece_params)
        SentencePieceTrainer.train(sentencepiece_params)
        #os.remove(self.temp_file)

    def get_processor(self) :
        processor = SentencePieceProcessor()
        processor.bos_token = ''
        processor.eos_token = ''
        processor.pad_token = ''
        processor.unk_token = ''
        processor.load(f"{self.model}.model")
        return processor

    def get_vocab_size(self) :
        return self.get_processor().vocab_size()

    def encode_source(self, text, sp_processor):
        byte_pairs = sp_processor.encode_as_ids(text)
        enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.pad_id()] * (self.max_len - len(list(byte_pairs)))
        return enc_c[:self.max_len]

    def encode_target(self, text, sp_processor):
        byte_pairs = sp_processor.encode_as_ids(text)
        enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.eos_id()] + [sp_processor.pad_id()] * (self.max_len - len(list(byte_pairs)))
        return enc_c[:self.max_len]

    def build(self) : 
        df = self.preload()
        self.train()
        df.to_csv(self.build_file)

    