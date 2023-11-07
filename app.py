from src.dataset import create_dataset
from src.train import train
from src.predict import predict


fields = ["predict", "train", "create_dataset"]
for i, field in enumerate(fields):
    print(f"Line {i+1}: {field}")

inp = ''

while(1) :
    action_index = input(f"Your action is {inp}: ")
    if action_index.isdigit() :
        action = fields[int(action_index) - 1]
        match action:
            case "predict":
                predict()
            case "train":
                train()
            case "create_dataset":
                create_dataset()
            case _:
                print("There is no such action")
    else :
        print("Please, enter a valid number")


