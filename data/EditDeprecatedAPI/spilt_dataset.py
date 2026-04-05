import json
import random
def split_dataset(input_file, train_file, test_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    split_index = len(data) // 2
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    input_file = 'data/EditDeprecatedAPI/deepseek-1.3b/all.json'
    train_file = 'data/EditDeprecatedAPI/deepseek-1.3b/train.json'
    test_file = 'data/EditDeprecatedAPI/deepseek-1.3b/test.json'
    split_dataset(input_file, train_file, test_file)