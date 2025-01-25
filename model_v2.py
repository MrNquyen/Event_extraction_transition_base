import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
import nn_v2
import logging
import ops_v2 as ops
# from dy_utils import ParamManager as pm
# from dy_utils import AdamTrainer
from event_eval import EventEval
from io_utils import to_set, get_logger
from shift_reduce_v2 import ShiftReduce

# Assuming `read_yaml` and other utils are similar
from io_utils import read_yaml
joint_config = read_yaml('joint_config.yaml')
data_config = read_yaml('data_config.yaml')

# Set seeds for reproducibility
random.seed(joint_config['random_seed'])
np.random.seed(joint_config['random_seed'])
torch.manual_seed(joint_config['random_seed'])

# Setup Logger
logging.basicConfig(
    format='%(asctime)s %(message)s',
    filemode='w'
)
logger = logging.getLogger()

sent_vec_dim = 0
if joint_config['use_sentence_vec']:
    train_sent_file = data_config['train_sent_file']
    test_sent_file = data_config['test_sent_file']
    dev_sent_file = data_config['dev_sent_file']
    train_sent_arr = np.load(train_sent_file)
    dev_sent_arr = np.load(dev_sent_file)
    test_sent_arr = np.load(test_sent_file)
    sent_vec_dim = train_sent_arr.shape[1]
    joint_config['sent_vec_dim'] = sent_vec_dim
    logger.info('train_sent_arr shape:%s'%str(train_sent_arr.shape))



# Define the model architecture using PyTorch
class MainModel(nn.Module):
    def __init__(self, n_words, action_dict, ent_dict, tri_dict, arg_dict, pos_dict, pretrained_vec=None):
        super(MainModel, self).__init__()

        # Define model components
        self.sent_model = nn.Module()
        
        if not joint_config['use_pretrain_embed'] and not joint_config['use_sentence_vec']:
            raise AttributeError('At least one of use_pretrain_embed and use_sentence_vec should be True')

        if joint_config['use_pretrain_embed']:
            self.word_embed = nn_v2.Embedding(
                n_words,
                joint_config['word_embed_dim'],
                init_weight=pretrained_vec,
                trainable=joint_config['pretrain_embed_tune']
            )
            
        if joint_config['use_char_rnn']:
            self.char_embed = nn_v2.Embedding(
                joint_config['n_chars'],
                joint_config['char_embed_dim'],
                trainable=True
            )
            self.char_rnn = nn_v2.MultiLayerLSTM(joint_config['char_embed_dim'], joint_config['char_rnn_dim'], bidirectional=True)

        if joint_config['use_pos']:
            self.pos_embed = nn_v2.Embedding(len(pos_dict), joint_config['pos_embed_dim'])

        if joint_config['random_word_embed']:
            print('Random_word_embed: True')
            self.word_embed_tune = nn_v2.Embedding(n_words, joint_config['word_embed_dim'], trainable=True)
            self.word_linear = nn_v2.Linear(joint_config['word_embed_dim'] * 2, joint_config['word_embed_dim'], activation='relu')

        if joint_config['use_sentence_vec']:
            print('Use_sentence_vec (BERT): True')
            self.train_sent_embed = nn_v2.Embedding(
                train_sent_arr.shape[0],sent_vec_dim,
                init_weight=train_sent_arr,
                trainable=False,
                name='trainSentEmbed',
            )

            self.dev_sent_embed = nn_v2.Embedding(
                dev_sent_arr.shape[0], sent_vec_dim,
                init_weight=dev_sent_arr,
                trainable=False,
                name='devSentEmbed'
            )

            self.test_sent_embed = nn_v2.Embedding(
                test_sent_arr.shape[0], sent_vec_dim,
                init_weight=test_sent_arr,
                trainable=False,
                name='testSentEmbed',
            )


            if joint_config['sent_vec_project'] > 0:
                print('Sentence_vec project to', joint_config['sent_vec_project'])
                self.sent_project = nn_v2.Linear(
                    sent_vec_dim, joint_config['sent_vec_project'],
                    activation=joint_config['sent_vec_project_activation'])

        # For RNN Encoder
        rnn_input = 0
        rnn_input = 0  # + config['char_rnn_dim'] * 2
        if joint_config['use_pretrain_embed']:
            rnn_input += joint_config['word_embed_dim']
            print('use_pretrain_embed:', joint_config['use_pretrain_embed'])

        if joint_config['use_sentence_vec'] and not joint_config['cat_sent_after_rnn']:
            rnn_input += sent_vec_dim
            print('use_sentence_vec:', joint_config['use_sentence_vec'])

        if joint_config['use_pos']:
            rnn_input += joint_config['pos_embed_dim']
            print('use_pos:', joint_config['use_pos'])

        if joint_config['use_char_rnn']:
            rnn_input += joint_config['char_rnn_dim'] * 2
            print('use_char_rnn:', joint_config['use_char_rnn'])


        if joint_config['use_rnn_encoder']:
            self.encoder = nn_v2.MultiLayerLSTM(
                rnn_input, joint_config['rnn_dim'],
                n_layer=joint_config['encoder_layer'], bidirectional=True,
                dropout_x=joint_config['dp_state'], dropout_h=joint_config['dp_state_h']
            )

        self.encoder_output_dim = 0
        if joint_config['use_rnn_encoder']:
            self.encoder_output_dim += joint_config['rnn_dim'] * 2

        elif joint_config['use_pretrain_embed']:
            self.encoder_output_dim += joint_config['word_embed_dim']
            if joint_config['use_pos']:
                self.encoder_output_dim += joint_config['pos_embed_dim']

        if joint_config['cat_sent_after_rnn'] and joint_config['use_sentence_vec']:
            self.encoder_output_dim += sent_vec_dim

        if joint_config['encoder_project'] > 0:
            self.encoder_project = nn.Linear(self.encoder_output_dim, joint_config['encoder_project'])


        # Shift Reduce Parser Layer
        self.shift_reduce = ShiftReduce(
            joint_config, self.encoder_output_dim, action_dict, ent_dict, tri_dict, arg_dict
        )

        self.optimizer = optim.Adam(self.parameters(), lr=joint_config['init_lr'])

    def forward(self, toks, chars, act_ids, acts, tris, ents, args, sent_range, pos_list):
        context_emb, sent_vec = self.input_embed(toks, chars, pos_list, sent_range, return_sent_vec=True)
        print(f'context_emb, sent_vec type: {type(context_emb)} - {sent_vec}')

        log_prob_list, loss_roles, loss_rels, pred_ents, pred_tris, pre_args, pred_acts = \
            self.shift_reduce(
                toks, 
                context_emb, 
                sent_vec, 
                act_ids, 
                acts, 
                is_train=True,
                ents=ents, 
                tris=tris, 
                args=args
            )

        act_loss = -torch.sum(log_prob_list)
        role_loss = torch.sum(loss_roles) if loss_roles else 0
        rel_loss = torch.sum(loss_rels) if loss_rels else 0
        loss = act_loss + role_loss + 0.05 * rel_loss

        return loss
    
    def get_word_embed(self, toks, pos_list, is_train=True):
        tok_emb = self.word_embed(toks)

        if joint_config['random_word_embed']:
            tok_emb_tune = self.word_embed_tune(toks)
            tok_emb = ops.cat_list(tok_emb , tok_emb_tune)
            tok_emb = self.word_linear(tok_emb)

        return tok_emb


    def input_embed(
            self, toks, chars, pos_list, range, is_train=True,
            return_last_h=False, return_sent_vec=False, mtype='train'
        ):
        toks = torch.tensor(toks)
        print(f'Type Tokens: {type(toks)}')
        tok_emb = self.word_embed(toks)
        last_h = None
        output_elmo_emb = None
        # if joint_config['use_rnn_encoder']:
        #     self.encoder.init_sequence(not is_train)

        if joint_config['use_pretrain_embed']:
            tok_emb = self.get_word_embed(toks, pos_list,  is_train)
            if joint_config['cat_sent_after_rnn'] and joint_config['use_rnn_encoder']:
                tok_emb, (last_h, last_c) = self.encoder.last_step(tok_emb)
            print(f'use_pretrain_embed.shape {tok_emb[0].shape}')
       
        if joint_config['use_sentence_vec']:
            sent_vec, output_elmo_emb = self.get_sent_embed(range, is_train, mtype=mtype)
            if tok_emb is not None:
                tok_emb = ops.cat_list(tok_emb, sent_vec)
            else:
                tok_emb = sent_vec
            print(f'use_sentence_vec.shape {tok_emb[0].shape}')

        if joint_config['use_char_rnn']:
            char_embed = self.get_char_embed(chars, is_train)
            print(f'tok_emb: {len(tok_emb)}')

            pooled_char_emb = [torch.max(item, dim=1).values for item in char_embed]
            # pooled_char_emb = torch.ma
            type_pooled_char_emb = [type(item) for item in pooled_char_emb]
            print(f'type_pooled_char_emb: {type_pooled_char_emb}')
            # print(f'char_embed: {char_embed[0].shape}')
            # print(f'tok_emb.shape {tok_emb[0].shape}')
            tok_emb = ops.cat_list(tok_emb, pooled_char_emb, dim=0)

        if joint_config['use_pos']:
            pos_emb = self.pos_embed(pos_list)
            #if is_train:pos_emb = nn_v2.dropout_list(pos_emb, 0.2)
            tok_emb = ops.cat_list(tok_emb, pos_emb)

        if is_train:
            tok_emb = ops.dropout_list(tok_emb, joint_config['dp_emb'])

        if  not joint_config['cat_sent_after_rnn'] and joint_config['use_rnn_encoder']:
            tok_emb, (last_h, last_c) = self.encoder.last_step(tok_emb)

        if is_train:
            tok_emb = ops.dropout_list(tok_emb, joint_config['dp_rnn'])

        if return_sent_vec:
            return tok_emb, output_elmo_emb
        else:
            return tok_emb

    def get_char_embed(self, chars, is_train=True):
        # self.char_rnn.init_sequence(not is_train)
        encoder_char = []
        for word_char in chars:
            word_char = torch.tensor(word_char)
            # print(f'word_char type: {type(word_char)}')
            char_embed = self.char_embed(word_char)
            _, (last_h, last_c) = self.char_rnn.last_step(char_embed)
            encoder_char.append(last_h)
        # return torch.stack(encoder_char)
        return encoder_char

    def get_sent_embed(self, range, is_train=True, mtype='train'):
        range = torch.tensor(range)
        if mtype=='train':
            print(f'range type: {type(range)}')
            sent_emb = self.train_sent_embed(range)
            
        elif mtype =='dev':
            sent_emb = self.dev_sent_embed(range)
            
        else:
            sent_emb = self.test_sent_embed(range)
            
        if joint_config['sent_vec_project'] > 0:
            sent_emb = self.sent_project(sent_emb)

        return sent_emb, sent_emb

    def decode(
            self, toks, chars, act_ids, acts, tris, ents, args, sent_range, pos_list, mtype='dev'
        ):

        context_emb, sent_vec = self.input_embed(toks, chars, pos_list, sent_range)
        log_prob_list, loss_roles, loss_rels, pred_ents, pred_tris, pre_args, pred_acts =\
                                            self.shift_reduce(
                                                toks,
                                                context_emb, sent_vec, act_ids, acts,
                                                is_train=False,
                                                ents=ents, tris=tris, args=args
                                            )
        return 0, pred_ents, pred_tris, pre_args

    def iter_batch_data(self, batch_data):
        batch_size = len(batch_data['tokens_ids'])

        for i in range(batch_size):
            one_data = {name:val[i]  for name, val in batch_data.items()}
            yield one_data


    def decay_lr(self, rate):
        self.optimizer.learning_rate *= rate


    def get_lr(self):
        return self.optimizer.learning_rate


    def set_lr(self, lr):
        self.optimizer.learning_rate = lr


    def update(self):
        try:
            self.optimizer.update()
        except RuntimeError:
            pass


    def regularization_loss(self, coef=0.001):
        losses = []
        for name, param in self.sent_model.named_parameters():
            if name.startswith('linearW'):
                # Compute L2 norm (squared) for each parameter
                losses.append(torch.norm(param, p=2) ** 2)
        
        total_loss = torch.sum(torch.stack(losses))
        return (coef / 2) * total_loss


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        self.load_state_dict(torch.load(path))
