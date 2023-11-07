from .config import get_conf_section
import pandas as pd
import torch
from .dataset import get_vocab_size
from .transformer import TransformerBuilder, device
from .preprocessor import CustomIterableDatasetv1

def get_train_loader() :
    batch_size = 100
    source_path = 'dataset/translation_data.csv'
    df = pd.read_csv(source_path)
    length = len(df)
    ### No shuffle because it's an iterative loader
    return torch.utils.data.DataLoader(CustomIterableDatasetv1(source_path, length),
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


def train():
    train_loader = get_train_loader()

    config = get_conf_section("training")

    epochs = config["epochs"]
    d_model = config["d_model"]
    heads = config["heads"]
    num_layers = config["num_layers"]
    vocab_size = get_vocab_size()

    transformer, criterion, transformer_optimizer = TransformerBuilder(d_model, heads, num_layers, vocab_size).export()
    return
    for epoch in range(epochs):
        
        run_epoch(train_loader, transformer, criterion, transformer_optimizer, epoch)
        
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_big_' + str(epoch) + '.pth.tar')


def run_epoch(train_loader, transformer, criterion, transformer_optimizer, epoch):
    
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
     
