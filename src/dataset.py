

import pandas as pd
import random
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from constants import *

class Dataset :
    def preload(self) :
        random.seed(42)
        dff = pd.read_csv(SRC_FILE, chunksize=500000)
        df = next(dff)
        df = df.rename(columns={'Source words/sentences':'source','Target words/sentences':'target'})
        df.sample(10)
        df['source'] = df['source'].astype('str')
        df['target'] = df['target'].astype('str')
        sample_text = list(df['source'] + df['target'])
        with open(f'{DATA_DIR}/{input_file}', "w", encoding="utf-8") as output:
            output.write(str(("\n").join(sample_text)))

        return df    

    #=============================================


    def train(self) :
        sentencepiece_params = ' '.join([
            '--input={}'.format(f'{DATA_DIR}/{input_file}'),
            '--model_type={}'.format(model_type),
            '--model_prefix={}'.format(f'{SP_DIR}/{model_prefix}'),
            '--vocab_size={}'.format(sp_vocab_size),
            '--pad_id={}'.format(pad_id),
            '--unk_id={}'.format(unk_id),
            '--bos_id={}'.format(bos_id),
            '--eos_id={}'.format(eos_id)
        ])
        print(sentencepiece_params)
        SentencePieceTrainer.train(sentencepiece_params)

    #=============================================
    def get_processor(self) :
        sp = SentencePieceProcessor()
        sp.bos_token = ''
        sp.eos_token = ''
        sp.pad_token = ''
        sp.unk_token = ''
        sp.load(f"{SP_DIR}/{model_prefix}.model")
        return sp

    def encode_source(self, text, sp_processor):
        byte_pairs = sp_processor.encode_as_ids(text)
        enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.pad_id()] * (seq_max_len - len(list(byte_pairs)))
        return enc_c[:seq_max_len]

    def encode_target(self, text, sp_processor):
        byte_pairs = sp_processor.encode_as_ids(text)
        enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.eos_id()] + [sp_processor.pad_id()] * (seq_max_len - len(list(byte_pairs)))
        return enc_c[:seq_max_len]


    def build(self) :
        df = self.preload()
        df.tail()
        df_f = df.sample(frac = 1)
        df_f.head()
        df_f.to_csv(f'{DATA_DIR}/{COMPILED_FILE}')
        self.train()
     
