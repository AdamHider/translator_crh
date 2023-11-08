import torch
from .dataset import Dataset
from .transformer import device

def evaluate(transformer, english, english_mask, max_len):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    transformer.eval()
    ds = Dataset()
    processor = ds.get_processor()
    start_token = processor.bos_id()
    end_token = processor.eos_id()
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
    
    sentence = processor.decode_ids(sen_idx)
    
    return sentence
     
