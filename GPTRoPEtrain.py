from transformers import  GPT2Tokenizer, GPT2Config, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import random
from torch.utils.data import IterableDataset

from torch import float32, bfloat16

bfloat = bfloat16  # 定义 BF16 数据类型


def convert_to_bfloat16(module):
    for param in module.parameters():
        param.data = param.data.to(bfloat)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(bfloat)

# 在模型初始化之后调用



from transformers import  AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import random
from RoPEGPT2 import RoPEGPT2Model


from torch.utils.data import IterableDataset


config = GPT2Config(n_positions=512)
model = RoPEGPT2Model(config)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
new_eos_token = "<|endoftext|>"
tokenizer.add_tokens([new_eos_token], special_tokens=True)
tokenizer.eos_token = new_eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
class OptimizedIterableDataset(IterableDataset):
    def __init__(self, BIO_path=None, QA_path=None, tokenizer=None, batch_size=48):
        assert BIO_path is not None and QA_path is not None, "Paths to data files must be provided"
        assert tokenizer is not None, "Tokenizer must be provided"
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        # Flags to indicate if the end of the file has been reached
        self.BIO_eof = False
        self.QA_eof = False

        # Read and shuffle BIO lines
        with open(BIO_path, 'r', encoding='utf-8') as file:
            self.BIO_lines = file.readlines()
        random.shuffle(self.BIO_lines)
        self.BIO_lines = iter(self.BIO_lines)  # Convert back to an iterator for consistent usage

        # Read and shuffle QA lines
        with open(QA_path, 'r', encoding='utf-8') as file:
            self.QA_lines = file.readlines()
        self.QA_lines = self.QA_lines*7
        random.shuffle(self.QA_lines)
        self.QA_lines = iter(self.QA_lines) 
        
        
    def line_reader(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()

    def get_block(self, lines):
        block = []
        try:
            line = next(lines)
            while len(self.tokenizer.tokenize(' '.join(block + [line]))) < 512:
                block.append(line)
                line = next(lines)
        except StopIteration:
            pass  # End of file reached

        if block:
            block = self.tokenizer(' '.join(block), padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            block['labels'] = block['input_ids'].detach().clone()
            return block
        return None

    def __iter__(self):
        while not self.BIO_eof or not self.QA_eof:  # 假设已经有逻辑来设置这些EOF标志
            batch = []
            for _ in range(self.batch_size):
                # 假设get_block是生成数据块的方法，它可能返回None
                data_block = self.get_block(self.BIO_lines if random.random() <= 0.2 else self.QA_lines)
                
                if data_block is None:
                    self.BIO_eof = True if random.random() <= 0.2 else self.QA_eof  # 假设根据随机条件设置EOF标志
                    break  # 遇到None，立即终止循环

                batch.append(data_block)
            
            if not batch:  # 如果批次为空（因为遇到None而跳出循环）
                break  # 退出外层循环

            yield self.prepare_batch(batch)  # 假设prepare_batch是准备批次的方法

                # Check if both files have reached EOF

    def prepare_batch(self, batch):
        input_ids = torch.cat([b['input_ids'] for b in batch])
        attention_mask = torch.cat([b['attention_mask'] for b in batch])
        labels = torch.cat([b['labels'] for b in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

            

            
            
# Create an instance of the dataset
BIO_path = '/home/yy/recurrent/TrainData/V8/BIO_V6_2_end.txt'
QA_path = '/home/yy/recurrent/QA_V6_5_end.txt'


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
convert_to_bfloat16(model)
model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# Training loop
import os
from torch.cuda.amp import autocast, GradScaler

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaling_factor=0.8


from transformers import AdamW, get_cosine_schedule_with_warmup
import math

total_steps = 90000 
batch_size = 40 
num_epochs =100

initial_lr = 0.001
min_lr = 0.0001
num_warmup_steps = 1000
total_steps = math.ceil(total_steps / batch_size) * num_epochs
weight_decay = 0.1
eps = 1e-6

optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay, eps=eps)

# Create learning rate scheduler 
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                             num_warmup_steps=num_warmup_steps,
                                             num_training_steps=total_steps)



model.train()
for epoch in range(100):
    total_loss = 0
    dataset = OptimizedIterableDataset(BIO_path=BIO_path, QA_path=QA_path, tokenizer=tokenizer,batch_size=48)  
    temp=0
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=8,  # Number of worker processes for data loading
        pin_memory=True,  # Pin memory for faster GPU transfer
    )

    for batch in dataloader:
        temp += 1
        # Move each tensor in the batch to the correct device
        inputs = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        if temp % 100 == 0:
            print(f'GPTRoPE Epoch: {epoch}, Step: {temp}, Loss: {loss.item()}')
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / temp
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
    
    # Save the model and tokenizer periodically
    if epoch % 5 == 0:
        model_name = f'check_point_{epoch+1}'
        save_path = os.path.join('/data/yy/GPTRoPE_V1', model_name)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
