from .config import get_conf, get_int
import pandas as pd
import torch
from .dataset import Dataset
from .transformer import TransformerBuilder, device
from .dataset_iterator import DatasetIterator

class Trainer : 
        
    def __init__(self):
        self.savepath = get_conf("path", "training")
        self.data_source = f'{get_conf("path", "dataset")}/{get_conf("dataset", "final_filename")}'
        self.batch_size = get_int("training", "batch_size")
        self.epochs = get_int("training", "epochs")
        self.d_model = get_int("training", "d_model")
        self.heads = get_int("training", "heads")
        self.num_layers = get_int("training", "num_layers")
        ds = Dataset()
        self.vocab_size = ds.get_vocab_size()

    def get_data_loader(self) :
        df = pd.read_csv(self.data_source)
        length = len(df)
        return torch.utils.data.DataLoader(DatasetIterator(self.data_source, length), batch_size = int(self.batch_size), pin_memory = True)
            
    ### Function to create masks

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


    def train(self):
        data_loader = self.get_data_loader()
        transformer, transformer_optimizer, criterion = TransformerBuilder(self.d_model, self.heads, self.num_layers, self.vocab_size).export()
        for epoch in range(self.epochs):
            self.run_epoch(data_loader, transformer, criterion, transformer_optimizer, epoch)
            state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
            torch.save(state, f'{self.savepath}/checkpoint_' + str(epoch) + '.pth.tar')


    def run_epoch(self, data_loader, transformer, criterion, transformer_optimizer, epoch):
        
        transformer.train()
        sum_loss = 0
        count = 0

        for i, (source, target) in enumerate(data_loader):
            
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
            out = transformer(source, source_matrix, target_input, target_input_matrix)

            # Compute the loss
            loss = criterion(out, target_result, target_result_matrix)
            
            # Backprop
            transformer_optimizer.optimizer.zero_grad()
            loss.backward()
            transformer_optimizer.step()
            
            sum_loss += loss.item() * samples
            count += samples
            
            if i % 100 == 0:
                print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(data_loader), sum_loss/count))
     
