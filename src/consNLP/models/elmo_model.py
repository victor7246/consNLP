import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class SeqModel_Elmo(nn.Module):
    def __init__(self, num_labels, extractor_type,  hidden_dim, if_attn):
        super(SeqModel_Elmo, self).__init__()
        self.if_attn = if_attn
        self.elmoLayer = ElmoLayer(options_file, weight_file)
        self.featureEncoder = FeatureEncoder(input_dim=2048, extractor_type= extractor_type, hidden_dim =hidden_dim)
        if self.if_attn:
            self.attention = Attention(hidden_dim)
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )


    def forward(self, words):
        emb_sequence, mask = self.elmoLayer(words)
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] => [128, 50, 100]
        if self.if_attn:
            features, att_weights = self.attention(features, mask.float())
        else:
            att_weights = None
        scores = self.score_layer(features) # features shape: [batch_size, max_seq_len, hidden_dim] => [128, 50, 32]
        return scores, mask, att_weights  # score shape: [batch_size, max_seq_len, num_labels] => [128, 50, 3]


class ElmoLayer(nn.Module):
    def __init__(self,options_file, weight_file):
        super(ElmoLayer, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.3)


    def forward(self, words):
        character_ids = batch_to_ids(words)
        elmo_output = self.elmo(character_ids)
        elmo_representation = torch.cat(elmo_output['elmo_representations'], -1)
        mask = elmo_output['mask']

        return elmo_representation, mask


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
            feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim] => [128, 44, 32]
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
