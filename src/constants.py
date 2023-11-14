import torch

# Path or parameters for data
DATA_DIR = './data'
SP_DIR = f'{DATA_DIR}/sp'
COMPILED_FILE = 'dataset.csv'
SRC_FILE = f'{DATA_DIR}/en-fr.csv'

# Parameters for sentencepiece tokenizer
pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3
input_file = 'tmp_dataset.txt'
model_prefix = 'dataset'
sp_vocab_size = 10000
character_coverage = 1.0
model_type = 'bpe'
seq_max_len = 50

# Parameters for Transformer & training
CKPT_DIR = './saved_model'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 100
num_epochs = 11
num_heads = 8
num_layers = 3
d_model = 512
warmup_steps = 4000
drop_out_rate = 0.1