import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Wavenet(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class SeqModel(nn.Module):
    def __init__(self, embeddings, num_labels, extractor_type,  hidden_dim, if_att):
        super(SeqModel, self).__init__()
        self.if_att = if_att
        print("hidden dim: ", hidden_dim)
        self.wordEmbedding = EmbeddingLayer(embeddings)
        self.featureEncoder = FeatureEncoder(input_dim=embeddings.shape[1], extractor_type= extractor_type, hidden_dim =hidden_dim)
        if self.if_att:
            self.attention = Attention(hidden_dim)

        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )

    def forward(self, w_tensor, mask):
        emb_sequence = self.wordEmbedding(w_tensor)  # w_tensor shape: [batch_size, max_seq_len] 
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] 

        if self.if_att:
            att_output, att_weights = self.attention(features, mask.float())
            scores = self.score_layer(att_output) # features shape: [batch_size, max_seq_len, hidden_dim] 
        else:
            scores = self.score_layer(features)  # features shape: [batch_size, max_seq_len, hidden_dim] 
            att_weights = None
        return scores, att_weights # score shape: [batch_size, max_seq_len, num_labels] 


class EmbeddingLayer(nn.Module):
    def __init__(self, embeddings):
        super(EmbeddingLayer, self).__init__()

        self.word_encoder = nn.Sequential(
            nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False),
            nn.Dropout(0.3)
        )


    def forward(self, w_tensor):
        return self.word_encoder(w_tensor)


class SeqModelWithoutEmb(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_labels, extractor_type,  hidden_dim, if_att):
        super(SeqModel, self).__init__()
        self.if_att = if_att
        print("hidden dim: ", hidden_dim)
        self.wordEmbedding = EmbeddingLayer(embedding_dim, vocab_size)
        self.featureEncoder = FeatureEncoder(input_dim=embedding_dim, extractor_type= extractor_type, hidden_dim =hidden_dim)
        if self.if_att:
            self.attention = Attention(hidden_dim)

        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )

    def forward(self, w_tensor, mask):
        emb_sequence = self.wordEmbedding(w_tensor)  # w_tensor shape: [batch_size, max_seq_len] 
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] 

        if self.if_att:
            att_output, att_weights = self.attention(features, mask.float())
            scores = self.score_layer(att_output) # features shape: [batch_size, max_seq_len, hidden_dim] 
        else:
            scores = self.score_layer(features)  # features shape: [batch_size, max_seq_len, hidden_dim] 
            att_weights = None
        return scores, att_weights # score shape: [batch_size, max_seq_len, num_labels] 



class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(EmbeddingLayer, self).__init__()

        self.word_encoder = nn.Sequential(
            nn.Embedding(vocab_size,embedding_dim),
            nn.Dropout(0.3)
        )


    def forward(self, w_tensor):
        return self.word_encoder(w_tensor)


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, extractor_type, hidden_dim):
        super(FeatureEncoder, self).__init__()


        self.extractor_type = extractor_type
        self.hidden_dim = hidden_dim

        if self.extractor_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)

    def forward(self, sequences, mask):
        """
               :param sequences: sequence shape: [batch_size, seq_len, emb_dim] => [128, 44, 100]
               :param mask:
               :return:
        """
        if self.extractor_type == 'lstm':
            lengths = torch.sum(mask, 1) # sum up all 1 values which is equal to the lenghts of sequences
            lengths, order = lengths.sort(0, descending=True)
            recover = order.sort(0, descending=False)[1]

            sequences = sequences[order]
            packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
            lstm_out, hidden = self.lstm(packed_words, None)

            feats, _ = pad_packed_sequence(lstm_out)
            feats = feats.permute(1, 0, 2)
            feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim] 
        return feats


class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float() * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

