from .config import get_conf
import pandas as pd
import random
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

DS_PATH = get_conf("path", "dataset")

MODEL_NAME = get_conf("dataset", "model_name")
INPUT_DS = get_conf("dataset", "input_dataset_name")
TMP_DS = get_conf("dataset", "temp_dataset_name")
FINAL_DS = get_conf("dataset", "final_dataset_name")

MAX_LEN = get_conf("dataset", "max_len")

def import_csv():
    source_path = f"{DS_PATH}/{INPUT_DS}"
    random.seed(42)
    df = pd.read_csv(source_path, header = None, chunksize = 500000)
    df = next(df)
    df[0] = df[0].astype('str')
    df[1] = df[1].astype('str')
    tmp_dataset = list(df[0] + df[1])
    with open(f"{DS_PATH}/{TMP_DS}", "w", encoding = "utf-8") as output:
        output.write(str(("\n").join(tmp_dataset)))
    return df

def train_sentencepiece():
    sentencepiece_params = ' '.join([
        '--input={}'.format(f"{DS_PATH}/{TMP_DS}"),
        '--model_type={}'.format("bpe"),
        '--model_prefix={}'.format(f"{DS_PATH}/{MODEL_NAME}"),
        '--vocab_size={}'.format(10000),
        '--pad_id={}'.format(0),
        '--unk_id={}'.format(1),
        '--bos_id={}'.format(2),
        '--eos_id={}'.format(3)
    ])
    print(sentencepiece_params)
    SentencePieceTrainer.train(sentencepiece_params)

#=============================================
def get_sp() :
    sp_processor = SentencePieceProcessor()
    sp_processor.bos_token = ''
    sp_processor.eos_token = ''
    sp_processor.pad_token = ''
    sp_processor.unk_token = ''
    sp_processor.load(f"{DS_PATH}/{MODEL_NAME}.model")
    return sp_processor

def get_vocab_size() :
    return get_sp().vocab_size()


def encode_source(text, sp_processor):
    byte_pairs = sp_processor.encode_as_ids(text)
    enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.pad_id()] * (MAX_LEN - len(list(byte_pairs)))
    return enc_c[:MAX_LEN]

def encode_target(text, sp_processor):
    byte_pairs = sp_processor.encode_as_ids(text)
    enc_c = [sp_processor.bos_id()] + list(byte_pairs) + [sp_processor.eos_id()] + [sp_processor.pad_id()] * (MAX_LEN - len(list(byte_pairs)))
    return enc_c[:MAX_LEN]

def create_dataset() : 
    df = import_csv()
    train_sentencepiece()
    df.to_csv(f"{DS_PATH}/{FINAL_DS}")