import pandas as pd
from dataset import Dataset
from transformer import *
from dataset_iterator import * 
from constants import *
import torch
import pandas as pd

class Trainer :

    def __init__(self) : 
        ds = Dataset()
        ds.build() 
        processor = ds.get_processor()
        source_path = f'{DATA_DIR}/{COMPILED_FILE}'
        df = pd.read_csv(source_path)
        length = len(df)
        self.train_loader = torch.utils.data.DataLoader(DatasetIterator(source_path, length, processor), batch_size = batch_size, pin_memory=True)
        vocab_size = processor.vocab_size()
        self.transformer = Transformer(d_model = d_model, heads = num_heads, num_layers = num_layers)
        self.transformer = self.transformer.to(device)
        adam_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = warmup_steps, optimizer = adam_optimizer)
        self.criterion = LossWithLS(vocab_size, drop_out_rate)
            
     
    def get_matrix(self, source, target_input, target_result):
        
        def subsequent_mask(size):
            mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
            return mask.unsqueeze(0)
        
        source_matrix = source!=0   ## This makes a matrix of true and falses
        source_matrix = source_matrix.to(device)
        source_matrix = source_matrix.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)

        target_input_matrix = target_input!=0
        target_input_matrix = target_input_matrix.unsqueeze(1)  # (batch_size, 1, max_words)
        target_input_matrix = target_input_matrix & subsequent_mask(target_input.size(-1)).type_as(target_input_matrix.data) 
        target_input_matrix = target_input_matrix.unsqueeze(1) # (batch_size, 1, max_words, max_words)
        target_result_matrix = target_result!=0              # (batch_size, max_words)
        
        return source_matrix, target_input_matrix, target_result_matrix
        
    
    def run_epoch(self, epoch):
        
        self.transformer.train()
        sum_loss = 0
        count = 0

        for i, (source, target) in enumerate(self.train_loader):
            
            samples = source.shape[0]

            # Move to device
            source = source.to(device)
            target = target.to(device)

            # Prepare Target Data
            target_input = target[:, :-1] ## Remove the  token
            target_result = target[:, 1:] ## Remove the  token

            # Create mask and add dimensions
            source_matrix, target_input_matrix, target_result_matrix = self.get_matrix(source, target_input, target_result)

            # Get the transformer outputs
            out = self.transformer(source, source_matrix, target_input, target_input_matrix)

            # Compute the loss
            loss = self.criterion(out, target_result, target_result_matrix)
            
            # Backprop
            self.transformer_optimizer.optimizer.zero_grad()
            loss.backward()
            self.transformer_optimizer.step()
            
            sum_loss += loss.item() * samples
            count += samples
            
            if i % 100 == 0:
                print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(self.train_loader), sum_loss/count))
        

    def train(self):
        for epoch in range(num_epochs):
            self.run_epoch(epoch)
            state = {'epoch': epoch, 'transformer': self.transformer, 'transformer_optimizer': self.transformer_optimizer}
            torch.save(state, f'{CKPT_DIR}/checkpoint_big_' + str(epoch) + '.pth.tar')
        
