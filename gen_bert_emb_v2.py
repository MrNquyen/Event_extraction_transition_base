import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel
from io_utils import read_yaml, read_json_lines

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read configuration and data
data_config = read_yaml('data_config.yaml')
data_dir = data_config['data_dir']
ace05_event_dir = data_config['ace05_event_dir']

train_list = read_json_lines(os.path.join(ace05_event_dir, 'train_nlp_ner.json'))
dev_list = read_json_lines(os.path.join(ace05_event_dir, 'dev_nlp_ner.json'))
test_list = read_json_lines(os.path.join(ace05_event_dir, 'test_nlp_ner.json'))

train_sent_file = data_config['train_sent_file']

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def save_bert(inst_list, filter_tri=True, name='train'):
    sents = []
    sent_lens = []
    for inst in inst_list:
        words, trigger_list, ent_list, arg_list = inst['nlp_words'], inst['Triggers'], inst['Entities'], inst['Arguments']
        # Empirically filter out sentences where event size is 0 or entity size less than 3 (for training)
        if len(trigger_list) == 0 and len(ent_list) < 3 and filter_tri: continue
        sents.append(words)
        sent_lens.append(len(words))

    max_length = 32
    table_length = 32 * len(inst_list)
    total_word_nums = sum(sent_lens)
    # input_table = np.empty((total_word_nums, 768))  # BERT output dimension (768 for base model)
    input_table = np.empty((table_length, 768))  # BERT output dimension (768 for base model)
    acc_len = 0
    print(f'input_table shape: {input_table.shape}')
    print(f'num sents: {len(sents)}')
    print(f'num total_word_nums: {total_word_nums}')
    for i, words in enumerate(sents):
        print(f'Words: {words}')
        if i % 100 == 0:
            print(f'progress: {i}, {len(sents)}')
        sent_len = sent_lens[i]
        
        # Tokenize the sentence using the Hugging Face tokenizer
        encodings = tokenizer(
            ' '.join(words), 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt',
            max_length=max_length,
        )
        input_ids = encodings['input_ids'].to(device)

        # Get the BERT embeddings from the model
        with torch.no_grad():
            outputs = bert_model(input_ids)
            embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        print(f'embeddings.shape {embeddings.shape}')
        print(f'len embedding {len(embeddings[0])}')
        for j, token_embedding in enumerate(embeddings[0]):
            start = acc_len + j
            # print(start)
            input_table[start, :] = token_embedding.cpu().detach().numpy()
        acc_len += sent_len

    # Determine the correct file name based on the data split
    bert_fname = data_config['train_sent_file'] if name == 'train' else \
                 data_config['dev_sent_file'] if name == 'dev' else data_config['test_sent_file']
    np.save(bert_fname, input_table)

    print('total_word_nums:', total_word_nums)

if __name__ == "__main__":
    save_bert(train_list, name='train')
    save_bert(dev_list, filter_tri=False, name='dev')
    save_bert(test_list, filter_tri=False, name='test')
