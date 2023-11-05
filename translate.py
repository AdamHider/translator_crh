import torch

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