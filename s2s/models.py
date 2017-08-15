import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from utils.vocab import Constants


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, attn_input, context):
        """
        attn_input: batch x dim
        context: batch x sourceL x dim
        """
        # print(attn_input.size(), context.size())
        target = self.linear_in(attn_input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        context_combined = torch.cat((weighted_context, attn_input), 1)

        context_output = self.tanh(self.linear_out(context_combined))

        return context_output, attn


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, net_input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(net_input, (h_0[i], c_0[i]))
            net_input = h_1_i
            if i + 1 != self.num_layers:
                net_input = self.dropout(net_input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return net_input, (h_1, c_1)


class RNNEncoder(nn.Module):
    def __init__(self, config):
        super(RNNEncoder, self).__init__()

        self.layers = config['num_layers']
        self.num_directions = 2 if config['bidirectional'] else 1
        assert config['rnn_hidden_size'] % self.num_directions == 0
        self.hidden_size = config['rnn_hidden_size'] // self.num_directions
        self.input_size = config['word_emb_size']

        self.embedding = nn.Embedding(config['src_vocab_size'],
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size,
                           num_layers=config['num_layers'],
                           dropout=config['dropout'],
                           bidirectional=config['bidirectional'])

    def load_pretrained_vectors(self, config):
        if config['pre_word_embs_enc'] is not None:
            pretrained = torch.load(config['pre_word_embs_enc'])
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, enc_input, hidden=None):
        if isinstance(enc_input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = enc_input[1].data.view(-1).tolist()
            emb = pack(self.embedding(enc_input[0]), lengths)
        else:
            emb = self.embedding(enc_input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(enc_input, tuple):
            outputs = unpack(outputs)[0]
        return outputs, hidden_t


class AttnDecoder(nn.Module):
    def __init__(self, config):
        super(AttnDecoder, self).__init__()

        self.layers = config['num_layers']
        self.input_feed = config['input_feed']
        input_size = config['word_emb_size']

        if self.input_feed:
            input_size += config['rnn_hidden_size']

        self.embedding = nn.Embedding(config['trg_vocab_size'],
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)
        self.rnn = StackedLSTM(self.layers, input_size,
                               config['rnn_hidden_size'],
                               config['dropout'])

        self.attn = GlobalAttention(config['rnn_hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

        self.hidden_size = config['rnn_hidden_size']

    def load_pretrained_vectors(self, config):
        if config['pre_word_embs_dec'] is not None:
            pretrained = torch.load(config['pre_word_embs_dec'])
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, dec_input, hidden, context, init_output):
        emb = self.embedding(dec_input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        attn = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class Seq2SeqDialogModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqDialogModel, self).__init__()
        self.encoder = RNNEncoder(config)
        self.decoder = AttnDecoder(config)
        self.generator = nn.Linear(config['rnn_hidden_size'],
                                   config['trg_vocab_size'])

    def init_dec_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, src_input, trg_input):
        context, enc_hidden = self.encoder(src_input)
        init_output = self.init_dec_output(context)

        enc_hidden = (self.fix_enc_hidden(enc_hidden[0]),
                      self.fix_enc_hidden(enc_hidden[1]))

        dec_output, dec_hidden, _attn = self.decoder(trg_input, enc_hidden, context, init_output)

        return dec_output
