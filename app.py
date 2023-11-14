import sys
sys.path.insert(0, './src/')

from src.trainer import Trainer
from src.predictor import Predictor


fields = ["predict", "train"]
for i, field in enumerate(fields):
    print(f"Line {i+1}: {field}")

inp = ''

while(1) :
    action_index = input(f"Your action is {inp}: ")
    if action_index.isdigit() :
        action = fields[int(action_index) - 1]
        match action:
            case "predict":
                pr = Predictor()
                while(1):
                    text = input("English: ") 
                    if text == 'quit':
                        break
                    prediction = pr.predict(text)
                    print(prediction+'\n')
            case "train":
                print('train')
                tr = Trainer()
                tr.train()
            case _:
                print("There is no such action")
    else :
        print("Please, enter a valid number")