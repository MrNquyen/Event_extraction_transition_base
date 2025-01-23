import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation functions
def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def penalized_tanh(x):
    alpha = 0.25
    return torch.max(torch.tanh(x), alpha * torch.tanh(x))

def relu(x):
    return F.relu(x)

def cube(x):
    return x ** 3

def relu_list(rep_list):
    return [relu(x) for x in rep_list]

def elu(x, alpha=1.0):
    return F.elu(x, alpha=alpha)

def log_sigmoid(x):
    return torch.log(torch.sigmoid(x))

def softmax(x, dim=0):
    return F.softmax(x, dim=dim)

def log_softmax(x):
    return F.log_softmax(x, dim=-1)


class Linear(nn.Module):
    def __init__(self, n_in, n_out, bias=True, activation='linear', init_w=None):
        super(Linear, self).__init__()
        self.W = nn.Parameter(torch.randn(n_out, n_in) * 0.01) if init_w is None else nn.Parameter(init_w)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.activation = activation
    
    def forward(self, input):
        output = torch.matmul(input, self.W.t()) + (self.bias if self.bias is not None else 0)
        if self.activation == 'linear':
            return output
        elif self.activation == 'sigmoid':
            return sigmoid(output)
        elif self.activation == 'tanh':
            return tanh(output)
        elif self.activation == 'relu':
            return relu(output)
        elif self.activation == 'elu':
            return elu(output)
        raise ValueError(f'Unknown activation function: {self.activation}')


class Embedding(nn.Module):
    def __init__(self, n_vocab, n_dim, init_weight=None, trainable=True, name='embed'):
        super(Embedding, self).__init__()
        self.trainable = trainable
        if init_weight is not None:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(init_weight), freeze=not trainable)
        else:
            self.embed = nn.Embedding(n_vocab, n_dim)

    def forward(self, input):
        return self.embed(input)
    
    def __getitem__(self, input):
        return self.embed(input)


import torch
import torch.nn as nn
import torch.optim as optim

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_x=0.0, dropout_h=0.0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_x = dropout_x
        self.dropout_h = dropout_h

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.kaiming_uniform_(self.weight_hh)

    def forward(self, x, hx, cx):
        if cx is None:
            cx = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        if x == None:
            x = torch.ones(x.shape).to(torch.float32)
        if self.training and self.dropout_x > 0.0:
            x = F.dropout(x, p=self.dropout_x, training=self.training)
            # x = x.to(torch.float32)

        if hx == None:
            hx = torch.ones(x.shape).to(torch.float32)
        if self.training and self.dropout_h > 0.0:
            hx = F.dropout(hx, p=self.dropout_h, training=self.training)

        print(f'x shape: {x.shape}')
        print(f'hx shape: {hx.shape}')
        print(f'self.weight_ih shape: {self.weight_ih.shape}')
        print(f'self.weight_hh shape: {self.weight_hh.shape}')

        x = x.to(torch.float32)
        hx = hx.to(torch.float32)
        if x.shape[0] != self.weight_ih.shape[1]:
            if len(x.shape) == 1:
                x_in_dim = x.size(0)
            else:
                x_in_dim = x.size(1)
            linear_x = nn.Linear(x_in_dim, self.weight_ih.size(1))  # Map input to output size
            x = linear_x(x)
        if hx.shape[0] != self.weight_hh.shape[1]:
            if len(hx.shape) == 1:
                hx_in_dim = hx.size(0)
            elif hx.shape[1] == self.weight_hh.shape[1]:
                hx_in_dim = hx.size(1)
            else:
                hx_in_dim = hx.size(0)

            linear_hx = nn.Linear(hx_in_dim, self.weight_hh.size(1))  # Map hidden state to output size
            hx = linear_hx(hx)
        # print(f'x_new shape: {x.shape}')
        # print(f'hx_new shape: {hx.shape}')
        gates = F.linear(x, self.weight_ih) + F.linear(hx, self.weight_hh) + self.bias

        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)

        cy = f * cx + i * g
        hy = o * torch.tanh(cy)

        return hy, cy

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False, dropout_x=0.0, dropout_h=0.0):
        super(MultiLayerLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.n_layer = n_layer

        self.fwd_layers = nn.ModuleList([LSTMCell(input_size, hidden_size, dropout_x, dropout_h)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([LSTMCell(input_size, hidden_size, dropout_x, dropout_h)])

        hidden_input_size = hidden_size * 2 if bidirectional else hidden_size
        for _ in range(1, n_layer):
            self.fwd_layers.append(LSTMCell(hidden_input_size, hidden_size))
            if bidirectional:
                self.bwd_layers.append(LSTMCell(hidden_input_size, hidden_size))

    def forward(self, inputs, hx=None, cx=None):
        layer_rep = inputs
        if self.bidirectional:
            fs, bs = None, None
            for fw, bw in zip(self.fwd_layers, self.bwd_layers):
                fs, _ = self._transduce_layer(fw, layer_rep, hx, cx)
                bs, _ = self._transduce_layer(bw, reversed(layer_rep), hx, cx)
                layer_rep = [torch.cat([f, b], dim=-1) for f, b in zip(fs, reversed(bs))]

            return layer_rep
        else:
            for fw in self.fwd_layers:
                layer_rep, _ = self._transduce_layer(fw, layer_rep, hx, cx)
            return layer_rep

    def _transduce_layer(self, layer, inputs, hx, cx):
        outputs, cells = [], []
        for x in inputs:
            hx, cx = layer(x, hx, cx)
            outputs.append(hx)
            cells.append(cx)

        return outputs, cells

    def last_step(self, inputs, hx=None, cx=None, separated_fw_bw=False):
        layer_rep = inputs
        if self.bidirectional:
            for fw, bw in zip(self.fwd_layers[:-1], self.bwd_layers[:-1]):
                fs, _ = self._transduce_layer(fw, layer_rep, hx, cx)
                bs, _ = self._transduce_layer(bw, reversed(layer_rep), hx, cx)
                layer_rep = [torch.cat([f, b], dim=-1) for f, b in zip(fs, reversed(bs))]

            fw, bw = self.fwd_layers[-1], self.bwd_layers[-1]
            fs, fc = self._transduce_layer(fw, layer_rep, hx, cx)
            bs, bc = self._transduce_layer(bw, reversed(layer_rep), hx, cx)
            layer_rep = [torch.cat([f, b], dim=-1) for f, b in zip(fs, reversed(bs))]

            last_rep = torch.cat([fs[-1], bs[-1]], dim=-1)
            last_c = torch.cat([fc[-1], bc[-1]], dim=-1)

            if not separated_fw_bw:
                return layer_rep, (last_rep, last_c)
            else:
                return (fw, bw), (last_rep, last_c)
        else:
            for fw in self.fwd_layers:
                layer_rep, _ = self._transduce_layer(fw, layer_rep, hx, cx)
            return layer_rep, (hx, cx)

class LayerNorm(nn.Module):
    def __init__(self, n_hid):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_hid, eps=1e-6)

    def transform(self, x):
        return self.layer_norm(x)

    def forward(self, input):
        if isinstance(input, list):
            return [self.transform(x) for x in input]
        else:
            return self.transform(input)


import torch
import torch.nn as nn

class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_x=0., dropout_h=0.):
        super(StackLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)  # Using PyTorch's LSTMCell
        self.empty_embedding = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)  # Equivalent to model.add_parameters in Dynet
        self.states = []
        self.indices = []

    def init_sequence(self, test=False):
        # PyTorch handles LSTMCell state initialization automatically in forward pass
        pass

    def get_reverse_hx(self):
        rev_hx = []
        for i in range(len(self.states) - 1, -1, -1):
            rev_hx.append(self.states[i][0])
        return rev_hx

    def iter(self):
        for (hx, cx), idx in zip(self.states, self.indices):
            yield hx, idx

    def push(self, input, idx):
        '''
        :param input: Tensor input to LSTMCell
        :param idx: word idx in buffer or action_id in vocab
        :return:
        '''
        if len(self.states) == 0:
            init_h = torch.zeros((input.size(0), self.hidden_size), device=input.device)  # Ensure correct device (CPU/GPU)
            init_c = torch.zeros((input.size(0), self.hidden_size), device=input.device)
            hx, cx = self.cell(input, (init_h, init_c))
        else:
            pre_hx, pre_cx = self.states[-1]
            hx, cx = self.cell(input, (pre_hx, pre_cx))

        self.states.append((hx, cx))
        self.indices.append(idx)

    def pop(self):
        if len(self.states) == 0:
            raise RuntimeError('Empty states')
        hx, cx = self.states.pop()
        idx = self.indices.pop()
        return hx, idx

    def last_state(self):
        return self.states[-1][0], self.indices[-1]

    def all_h(self):
        return [s[0] for s in self.states]

    def clear(self):
        self.states.clear()
        self.indices.clear()

    def embedding(self):
        if len(self.states) == 0:
            hx = self.empty_embedding
        else:
            hx, cx = self.states[-1]
        return hx

    def is_empty(self):
        return len(self.states) == 0

    def idx_range(self):
        return self.indices[0], self.indices[-1]

    def last_idx(self):
        return self.indices[-1]

    def __getitem__(self, item):
        hx, cx = self.states[item]
        idx = self.indices[item]
        return hx, idx

    def __len__(self):
        return len(self.states)

    def __str__(self):
        return str(len(self.states)) + ':' + str(len(self.indices))


class Buffer(object): 

    def __init__(self, bi_rnn_dim, hidden_state_list):
        '''
        :param hidden_state_list: list of tensors (each of shape (n_dim,))
        '''
        self.hidden_states = hidden_state_list
        self.seq_len = len(hidden_state_list)
        self.idx = 0

    def pop(self):
        if self.idx == self.seq_len:
            raise RuntimeError('Empty buffer')
        hx = self.hidden_states[self.idx]
        cur_idx = self.idx
        self.idx += 1
        return hx, cur_idx

    def last_state(self):
        return self.hidden_states[self.idx], self.idx

    def buffer_idx(self):
        return self.idx

    def hidden_embedding(self):
        return self.hidden_states[self.idx]

    def hidden_idx_embedding(self, idx):
        return self.hidden_states[idx]

    def is_empty(self):
        return (self.seq_len - self.idx) == 0

    def move_pointer(self, idx):
        self.idx = idx

    def move_back(self):
        self.idx -= 1

    def __len__(self):
        return self.seq_len - self.idx

class LambdaVar(object):
    TRIGGER = 't'
    ENTITY = 'e'
    OTHERS = 'o'

    def __init__(self, bi_rnn_dim):
        self.var = None
        self.idx = -1
        self.bi_rnn_dim = bi_rnn_dim
        self.lmda_empty_embedding = torch.zeros(bi_rnn_dim)  # Equivalent to dy.zeros in the original code
        self.lambda_type = LambdaVar.OTHERS

    def push(self, embedding, idx, lambda_type):
        self.var = embedding
        self.idx = idx
        self.lambda_type = lambda_type

    def pop(self):
        var, idx = self.var, self.idx
        self.var, self.idx = None, -1
        self.lambda_type = LambdaVar.OTHERS
        return var, idx

    def clear(self):
        self.var, self.idx = None, -1

    def is_empty(self):
        return self.var is None

    def is_trigger(self):
        return self.lambda_type == LambdaVar.TRIGGER

    def is_entity(self):
        return self.lambda_type == LambdaVar.ENTITY

    def embedding(self):
        # Return empty embedding if var is None, otherwise return the actual variable
        return self.lmda_empty_embedding if self.var is None else self.var