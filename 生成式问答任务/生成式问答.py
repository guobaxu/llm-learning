import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW, get_scheduler
# from evaluate import load
from tqdm.auto import tqdm
import sacrebleu


# 参数设置
max_input_length = 512
max_target_length = 64
train_batch_size = 4
test_batch_size = 4
learning_rate = 2e-5
epoch_num = 3
beam_size = 4
no_repeat_ngram_size = 2

# 设置随机数种子，确保结果可复现
seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')

# # 使用transformers的T5Tokenizer加载分词器
# # 这个分词器需要sentencepiece库，没有的话需要pip安装下
tokenizer = T5Tokenizer.from_pretrained("./mengzi-t5-base")
model = T5ForConditionalGeneration.from_pretrained("./mengzi-t5-base")
model = model.to(device)

# 数据集加载函数，继承自Dataset类
class DuReaderQG(Dataset):
    def __init__(self, datafile):
        self.data = self.load_data(datafile)

    def load_data(self, datafile):
        data = []
        with open(datafile, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                data.append(json_data)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
train_data = DuReaderQG("./DuReaderQG/train.json")
valid_data = DuReaderQG("./DuReaderQG/dev.json")
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')

# 定义批处理函数
def collote_fn(batch_samples):
    batch_inputs, batch_question, batch_targets = [], [], []
    for sample in batch_samples:
        batch_inputs.append(sample['context'])
        batch_question.append(sample['question'])
        batch_targets.append(sample['answer'])
    batch_data = tokenizer(
        batch_inputs,
        batch_question,
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, collate_fn=collote_fn)

# 定义训练函数，接口：dataloader，model，optimizer，lr_scheduler，
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        # 内置的loss
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def compute_metrics(preds, labels):
    # preds,labels:list[]
    b4 = []
    for pred, label in zip(preds, labels):
        pred = [pred]
        references = [label]
        # 计算 BLEU 指标
        results = sacrebleu.corpus_bleu(predictions=pred, references=references)
        b4.append(results)
    return  sum(b4) / len(b4)

def test_loop(dataloader, model):
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    # 计算bleu指标,返回bleu4指标
    result = compute_metrics(preds, labels)
    print(f"Bleu-4: {result:>0.2f}\n")
    return result



optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    bleu4 = test_loop(valid_dataloader, model)
    print(bleu4)
    if bleu4 > best_bleu:
        best_bleu = bleu4
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_bleu-4_{bleu4:0.4f}_model_weights.bin')
print("Done!")