import pandas as pd
import torch
from dataset import get_sp
from transformer import Transformer, AdamWarmup, LossWithLS, device
from preprocessor import CustomIterableDatasetv1


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
     

epochs = 8
d_model = 512
heads = 8
num_layers = 3
sp = get_sp()
vocab_size = sp.vocab_size()
    
transformer = Transformer(d_model = d_model, heads = heads, num_layers = num_layers)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
criterion = LossWithLS(vocab_size, 0.1)


def feed():
    for epoch in range(epochs):
        
        train(train_loader, transformer, criterion, epoch)
        
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_big_' + str(epoch) + '.pth.tar')


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
     
