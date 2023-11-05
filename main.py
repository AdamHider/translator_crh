
import pandas as pd
import random
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, IterableDataset
import pandas as pd
import random
import csv

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

input_file = 'sample_text.txt'
max_num_words = 10000
model_type = 'bpe'
model_prefix = 'sentencepiece'
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

sp = SentencePieceProcessor()
sp.bos_token = ''
sp.eos_token = ''
sp.pad_token = ''
sp.unk_token = ''
sp.load(f"{model_prefix}.model")
print('Found %s unique tokens.' % sp.get_piece_size())

#=============================================

original = "Je n'avais aucune idée"
encoded_pieces = sp.encode_as_pieces(original)
print(encoded_pieces)

# or convert it to numeric id for downstream modeling
encoded_ids = sp.encode_as_ids(original)
print(encoded_ids)
#'[3261, 45, 9924, 4030, 4298, 4774]'

#=============================================

decoded_pieces = sp.decode_pieces(encoded_pieces)
print(decoded_pieces)

# we can convert the numeric id back to the original text
decoded_ids = sp.decode_ids(encoded_ids)
print(decoded_ids)

#=============================================

max_len = 50

def encode_english(text, sp):
  byte_pairs = sp.encode_as_ids(text)
  enc_c = [sp.bos_id()] + list(byte_pairs) + [sp.pad_id()] * (max_len - len(list(byte_pairs)))
  return enc_c[:max_len]

sample_text = 'My name is Sean, and je suis un americain'
print(encode_english(sample_text, sp))

#=============================================

def encode_french(text, sp):
    byte_pairs = sp.encode_as_ids(text)
    enc_c = [sp.bos_id()] + list(byte_pairs) + [sp.eos_id()] + [sp.pad_id()] * (max_len - len(list(byte_pairs)))
    return enc_c[:max_len]

sample_text = 'My name is Sean, and je suis un americain'
print(encode_french(sample_text, sp))

#=============================================

df.tail()

#=============================================

df4 = df.sample(frac = 1)
df4.head()

#=============================================

df4.to_csv('translation_data.csv')
     

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
     

batch_size = 100
source_path = 'translation_data.csv'
df = pd.read_csv(source_path)
length = len(df)

### No shuffle because it's an iterative loader
train_loader = torch.utils.data.DataLoader(CustomIterableDatasetv1(source_path, length),
                                           batch_size = batch_size,  
                                           pin_memory=True)
     

### Function to create masks

def create_masks(english, french_input, french_target):
    
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    
    english_mask = english!=0   ## This makes a matrix of true and falses
    english_mask = english_mask.to(device)
    english_mask = english_mask.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)
  

    french_input_mask = french_input!=0
    french_input_mask = french_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    french_input_mask = french_input_mask & subsequent_mask(french_input.size(-1)).type_as(french_input_mask.data) 
    french_input_mask = french_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    french_target_mask = french_target!=0              # (batch_size, max_words)
    
    return english_mask, french_input_mask, french_target_mask
     

#=============================================
#BUILDING TRANSFORMER

class Embeddings(nn.Module):
    """
    Initializes embeddings
    Adds positional encoding
    """
    def __init__(self, vocab_size, d_model, max_len = 50):
        super(Embeddings, self).__init__()
        self.d_model = d_model  ## This is basically how deep the model is, so the embeddings are going to be 512 numbers for each word.
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1) 
        
    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):   # for each position of the word
            for i in range(0, d_model, 2):   # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # include the batch size
        return pe
        
    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model) ## To add more weight to embeddings
        embedding += self.pe[:, :embedding.size(1)]   # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding) ## Isn't this the same dropout?
        return embedding
     

class MultiHeadAttention(nn.Module):
    '''
    Runs Embeddings through linear layer to create query, keys, and values
    Divides q,k,v by heads
    '''
    def __init__(self, heads, d_model):
        
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, 512)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, 512) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0,1,3,2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)    # (batch_size, h, max_len, max_len)
        weights = F.softmax(scores, dim = -1)           # (batch_size, h, max_len, max_len)
        weights = self.dropout(weights)
        
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, h * d_k)
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        # (batch_size, max_len, h * d_k)
        interacted = self.concat(context)
        return interacted 
     

class FeedForward(nn.Module):
    '''
    This is a later linear layer, for adding more features.
    '''
    def __init__(self, d_model, middle_dim = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out
     

class EncoderLayer(nn.Module):
    '''
    creates encoder layer
    '''
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, mask):
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
     

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings, encoded, src_mask, target_mask):
        query = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded

     

class Transformer(nn.Module):
    
    def __init__(self, d_model, heads, num_layers):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = 10000
        self.embed = Embeddings(self.vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)
        
    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings
    
    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings
        
    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim = 2)
        return out
     

class AdamWarmup:
    
    def __init__(self, model_size, warmup_steps, optimizer):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()
     

class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size
        
    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))   # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)   # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)       # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)    # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss

     
d_model = 512
heads = 8
num_layers = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 8
vocab_size = sp.vocab_size()
    
transformer = Transformer(d_model = d_model, heads = heads, num_layers = num_layers)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
criterion = LossWithLS(vocab_size, 0.1)

def train(train_loader, transformer, criterion, epoch):
    
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (english, french) in enumerate(train_loader):
        
        samples = english.shape[0]

        # Move to device
        english = english.to(device)
        french = french.to(device)

        # Prepare Target Data
        french_input = french[:, :-1] ## Remove the  token
        french_target = french[:, 1:] ## Remove the  token

        # Create mask and add dimensions
        english_mask, french_input_mask, french_target_mask = create_masks(english, french_input, french_target)

        # Get the transformer outputs
        out = transformer(english, english_mask, french_input, french_input_mask)

        # Compute the loss
        loss = criterion(out, french_target, french_target_mask)
        
        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        
        sum_loss += loss.item() * samples
        count += samples
        
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))
     

def evaluate(transformer, english, english_mask, max_len):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    transformer.eval()
    start_token = sp.bos_id()
    end_token = sp.eos_id()
    encoded = transformer.encode(english, english_mask)
    words = torch.LongTensor([[start_token]]).to(device)
    
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, english_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == end_token:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)
        
    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()
        
    sen_idx = [w for w in words if w not in {start_token}]
    
    sentence = sp.decode_ids(sen_idx)
    
    return sentence
     
def feed():
    for epoch in range(epochs):
        
        train(train_loader, transformer, criterion, epoch)
        
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_big_' + str(epoch) + '.pth.tar')
     
#feed()

checkpoint = torch.load('checkpoint_big_7.pth.tar')
transformer = checkpoint['transformer']

while(1):
    english = input("English: ") 
    if english == 'quit':
        break
    english = encode_english(english, sp)
    english = torch.LongTensor(english).to(device).unsqueeze(0)
    english_mask = (english!=0).to(device).unsqueeze(1).unsqueeze(1)  
    sentence = evaluate(transformer, english, english_mask, int(40))
    print(sentence+'\n')