

import pandas as pd
import random
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor


def transform_csv():

    source_path = 'en-fr.csv'

    random.seed(42)

    dff = pd.read_csv(source_path, chunksize=500000)

    df = next(dff)

    df = df.rename(columns={'English words/sentences':'en','French words/sentences':'fr'})

    df.sample(10)
        
    df['en'] = df['en'].astype('str')
    df['fr'] = df['fr'].astype('str')
    sample_text = list(df['en'] + df['fr'])

    with open("sample_text.txt", "w", encoding="utf-8") as output:
        output.write(str(("\n").join(sample_text)))

#=============================================

model_prefix = 'sentencepiece'
def loadProcessor():
    input_file = 'sample_text.txt'
    max_num_words = 10000
    model_type = 'bpe'
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3
    sentencepiece_params = ' '.join([
        '--input={}'.format(input_file),
        '--model_type={}'.format(model_type),
        '--model_prefix={}'.format(model_prefix),
        '--vocab_size={}'.format(max_num_words),
        '--pad_id={}'.format(pad_id),
        '--unk_id={}'.format(unk_id),
        '--bos_id={}'.format(bos_id),
        '--eos_id={}'.format(eos_id)
    ])
    print(sentencepiece_params)
    SentencePieceTrainer.train(sentencepiece_params)

#=============================================
def get_sp() :
    sp = SentencePieceProcessor()
    sp.bos_token = ''
    sp.eos_token = ''
    sp.pad_token = ''
    sp.unk_token = ''
    sp.load(f"{model_prefix}.model")
    return sp


max_len = 50

def encode_english(text, sp):
  byte_pairs = sp.encode_as_ids(text)
  enc_c = [sp.bos_id()] + list(byte_pairs) + [sp.pad_id()] * (max_len - len(list(byte_pairs)))
  return enc_c[:max_len]

#=============================================

def encode_french(text, sp):
    byte_pairs = sp.encode_as_ids(text)
    enc_c = [sp.bos_id()] + list(byte_pairs) + [sp.eos_id()] + [sp.pad_id()] * (max_len - len(list(byte_pairs)))
    return enc_c[:max_len]


#=============================================

#df.tail()

#=============================================

#df4 = df.sample(frac = 1)
#df4.head()

#=============================================

#df4.to_csv('translation_data.csv')
     
