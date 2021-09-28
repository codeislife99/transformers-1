import os
from transformers import BertTokenizer
import torch

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def get_dataset_loader(batch_size, sequence_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, "clean.txt")

    os.system(f"wget -N https://raw.githubusercontent.com/jamescalam/transformers/main/data/text/meditations/clean.txt && mv clean.txt {dataset_path}")

    with open(dataset_path, 'r') as fp:
        text = fp.read().split('\n')

    print(text[:5])

    inputs = tokenizer(text, return_tensors='pt', max_length=sequence_length, truncation=True, padding='max_length')

    inputs['labels'] = inputs.input_ids.detach().clone()
    print(f"Input Keys:{inputs.keys()}")

    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
            (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = MeditationsDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader

use_xla = False
# model_list = ["bert-base-uncased", "bert-large-uncased", "/opt/ml/hopper/test/files/bart-config.json", "roberta-base", "gpt2"]
model_list = ["bert-base-uncased", "bert-large-uncased", "bart-config.json", "roberta-base", "gpt2"]
# model_list = ["bart-config.json"]
batch_size_list = [4,8]
# batch_size_list = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128]
# seq_len_list = [128, 512]
seq_len_list = [512]
for model in model_list:
    for seq_len in seq_len_list:
        for batch_size in batch_size_list:
            print("running {} batch_size={} sequence_length={}".format(model, batch_size, seq_len))
            # loader = get_dataset_loader(batch_size, seq_len)
            # torch.save(loader, '/pytorch/xla/test/loader.pt')
            # torch.save(loader, '/home/ubuntu/loader.pt')
            if use_xla: 
                os.system("python3 /home/ubuntu/hopper/transformers_david/examples/pytorch/benchmarking/run_benchmark.py --models {} --training yes --batch_sizes {} --sequence_lengths {} --inference no --tpu true --memory false --fp16 --dump_loss".format(model, batch_size, seq_len))            
            else:
                os.system("python3 /home/ubuntu/hopper/transformers_david/examples/pytorch/benchmarking/run_benchmark.py --models {} --training yes --batch_sizes {} --sequence_lengths {} --inference no --tpu false --memory false --fp16".format(model, batch_size, seq_len))
