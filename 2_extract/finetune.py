import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.optim import AdamW
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input_text = item["input"]
        output_text = item["output"]
        system = item["system"]
        normalized_value = item["normalized_value"]

        input_ids = tokenizer.encode(instruction + "\n" + input_text + "\n" + system, truncation=True, padding="max_length", max_length=2048)
        output_ids = tokenizer.encode(output_text, truncation=True, padding="max_length", max_length=2048)

        max_token_id = tokenizer.vocab_size

        input_ids = [id if id < max_token_id else 1 for id in input_ids]
        output_ids = [id if id < max_token_id else 1 for id in output_ids]

        # input_ids = [id if id < max_token_id else tokenizer.unk_token_id for id in input_ids]
        # output_ids = [id if id < max_token_id else tokenizer.unk_token_id for id in output_ids]


        target_prob = normalized_value
        return {
            "input_ids": torch.tensor(input_ids),
            "output_ids": torch.tensor(output_ids),
            "target_prob": torch.tensor(target_prob),
        }


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f) 
    return data




import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="finetune")
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Directory of the training data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the fine-tuned model"
    )
    
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for training (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-6, 
        help="Learning rate for optimizer (default: 5e-6)"
    )

    parser.add_argument(
        "--llama_dir", 
        type=str, 
        default="path/to/llama3-8b-instruct", 
        help="Directory of the Llama3-8b-instruct model"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f'llama dir: {args.llama_dir}')



    global device, model_name, tokenizer, model, optimizer   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model_name = args.llama_dir

    print(model_name)

    from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'eos_token': '[EOS]',
        'bos_token': '[BOS]'
    })
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({'unk_token': '[UNK]'})

    model = LlamaForCausalLM.from_pretrained(model_name).to(device) 
    model = model.to(device)

    # tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=False)
    # model = LlamaForCausalLM.from_pretrained(model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    data = load_data(f"{args.data_dir}/final/data_total.json")
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def kl_divergence(p, q):
        p = torch.tensor(p, dtype=torch.float32)
        q = torch.tensor(q, dtype=torch.float32)
        return 0.5 * torch.sum(p * torch.log(p / (q + 1e-8))) + 0.5 * torch.sum(q * torch.log(q / (p + 1e-8)))

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            output_ids = batch["output_ids"].to(device)
            target_prob = batch["target_prob"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, labels=output_ids)
            logits = outputs.logits

            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)

            
            target_prob_dist = [target_prob, 1 - target_prob] 
            target_prob_dist = torch.tensor(target_prob_dist).to(device)

            loss = kl_divergence(probs, target_prob_dist)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    model.save_pretrained(f"{args.output_dir}/finetuned_llama3-8b-instruct")
    tokenizer.save_pretrained(f"{args.output_dir}/finetuned_llama3-8b-instruct")