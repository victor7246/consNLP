from __future__ import absolute_import

import torch
import torch.nn as nn


from .layers import LearnableWeightedAvg, WeightedAvg, MultiSampleDropout, SCSE, MixLinear, SpatialDropout

class TransformerModel(nn.Module):
    def __init__(self, base_model, dropout=.1, n_out=1):
        super(TransformerModel, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0]
        o2 = torch.mean(o2, dim=1)
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class TransformerMNLI(nn.Module):
    def __init__(self, base_model, dropout=.1, n_out=1):
        super(TransformerMNLI, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids)
        o2 = o2[0]
        o2 = torch.mean(o2, dim=1)
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class TransformerModelSCSE(nn.Module):
    def __init__(self, base_model, max_len, r=8, hidden_size=48, dropout=.1, n_out=1):
        super(TransformerModelSCSE, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.SCSE = SCSE(max_len, r)
        self.linear = nn.Linear(base_model.config.hidden_size, hidden_size)
        self.out = nn.Linear(max_len * hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[2][1:]
        o2 = torch.cat([o2[i].expand((1,)+o2[i].shape) for i in range(self.base_model.config.num_hidden_layers)],axis=0)
        o2 = o2.transpose(0,1)
        o2 = o2.transpose(1,2)
        o2 = self.SCSE(o2)
        #print (o2.shape)
        o2= torch.mean(o2, dim=2)
        #print (o2.shape)
        o2 = self.linear(o2)

        batch_size, n_words, size = o2.shape
        o2 = torch.reshape(o2, (batch_size, n_words * size))
        o2 = self.drop(o2)
        output = self.out(o2)

        return output

class TransformerModelWithDropout2D(nn.Module):
    def __init__(self, base_model, hidden_size=100, dropout=.2, n_out=1):
        super(TransformerModelWithDropout2D, self).__init__()

        self.base_model = base_model
        self.drop1 = nn.Dropout2d(dropout)
        self.linear = nn.Linear(base_model.config.hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[2][1:]
        o2 = torch.cat([o2[i].expand((1,)+o2[i].shape) for i in range(self.base_model.config.num_hidden_layers)],axis=0)
        o2 = o2.transpose(0,1)
        o2 = self.drop1(o2)
        #print (o2.shape)
        o2 = torch.mean(o2, dim=2)
        #print (o2.shape)
        o2 = self.linear(o2)
        o2 = torch.mean(o2, dim=1)
        o2 = self.drop2(o2)
        output = self.out(o2)

        return output

class TransformerWithLSTM(nn.Module):
    def __init__(self, base_model, n_lstm_units=100, dropout=.2, n_out=1):
        super(TransformerWithLSTM, self).__init__()

        self.base_model = base_model
        self.dropout_p = dropout
        self.embedding_dropout = SpatialDropout(dropout)
        self.drop = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(base_model.config.hidden_size, n_lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(n_lstm_units * 2, n_lstm_units, bidirectional=True, batch_first=True)
        self.out = nn.Linear(n_lstm_units * 2,n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[2][0]
        o2 = self.embedding_dropout(o2)
        o2, _ = self.lstm1(o2)
        o2, _ = self.lstm2(o2)

        o2 = torch.mean(o2, dim=1)
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class TransformerWithCLS(nn.Module):
    def __init__(self, base_model, dropout=.1, n_out=1):
        super(TransformerWithCLS, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[1]
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class TransformerMNLIWithCLS(nn.Module):
    def __init__(self, base_model, dropout=.1, n_out=1):
        super(TransformerMNLIWithCLS, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids)
        o2 = o2[1]
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class MultiTransformer(nn.Module):
    def __init__(self, base_model1, base_model2, hidden_size=100, dropout=.1, n_out=1):
        super(MultiTransformer, self).__init__()

        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.out1 = nn.Linear(base_model1.config.hidden_size, hidden_size)
        self.out2 = nn.Linear(base_model2.config.hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_out)

    def calculate_softmax_weight(self, o2):
        weights = torch.cat([o2[i][:,0,:].expand((1,)+o2[i][:,0,:].shape) for i in range(12)],axis=0)
        weights = nn.Softmax(dim=0)(weights)
        weights = weights.expand((1,) + weights.shape)
        weights = weights.transpose(0,2)
        weights = weights.transpose(0,1)

        return weights
        
    def forward(self, ids1, ids2, mask1, mask2, token_type_ids1, token_type_ids2):
        o1 = self.base_model1(ids1, attention_mask=mask1, token_type_ids=token_type_ids1)[2][1:]
        o2 = self.base_model2(ids2, attention_mask=mask2, token_type_ids=token_type_ids2)[2][1:]

        self.weights1 = self.calculate_softmax_weight(o1)
        self.weights2 = self.calculate_softmax_weight(o2)

        o1 = WeightedAvg(self.weights1)(o1)
        o1 = torch.mean(o1, dim=1)
        o2 = WeightedAvg(self.weights2)(o2)
        o2 = torch.mean(o2, dim=1)

        o1 = self.drop1(o1)
        o2 = self.drop2(o2)

        o1 = self.out1(o1)
        o2 = self.out2(o2)

        out = o1 + o2
        output = self.out(out)

        return output

class MultiTransformerCLS(nn.Module):
    def __init__(self, base_model1, base_model2, hidden_size=100, dropout=.1, n_out=1):
        super(MultiTransformerCLS, self).__init__()

        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.out1 = nn.Linear(base_model1.config.hidden_size, hidden_size)
        self.out2 = nn.Linear(base_model2.config.hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_out)
        
    def forward(self, ids1, ids2, mask1, mask2, token_type_ids1, token_type_ids2):
        o1 = self.base_model1(ids1, attention_mask=mask1, token_type_ids=token_type_ids1)[1]
        o2 = self.base_model2(ids2, attention_mask=mask2, token_type_ids=token_type_ids2)[1]

        o1 = self.drop1(o1)
        o2 = self.drop2(o2)

        o1 = self.out1(o1)
        o2 = self.out2(o2)

        out = (o1 + o2)/2
        output = self.out(out)

        return output

class TransformerWithLayerwiseSoftmax(nn.Module):
    def __init__(self, base_model, dropout=.2, n_out=1, sample_wise_dropout=False, n_sample_dropout=4):
        super(TransformerWithLayerwiseSoftmax, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)
        self.sample_wise_dropout = sample_wise_dropout
        self.n_sample_dropout = n_sample_dropout

        if sample_wise_dropout:
            self.dropouts = MultiSampleDropout(n_sample_dropout,dropout)

    def calculate_softmax_weight(self, o2):
        weights = torch.cat([o2[i][:,0,:].expand((1,)+o2[i][:,0,:].shape) for i in range(self.base_model.config.num_hidden_layers)],axis=0)
        weights = nn.Softmax(dim=0)(weights)
        weights = weights.expand((1,) + weights.shape)
        weights = weights.transpose(0,2)
        weights = weights.transpose(0,1)

        self.weights = weights

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        #o2 = o2[0]
        #bo = o2[1]
        o2 = o2[2][1:]
        self.calculate_softmax_weight(o2)
        o2 = WeightedAvg(self.weights)(o2)
        
        bo = torch.mean(o2, dim=1)

        #bo = LearnableWeightedAvg(o2.shape[1])(o2)

        if self.sample_wise_dropout:
            output = 0
            for i in range(self.n_sample_dropout):
                output += self.out(self.dropouts.dropouts[i](bo)) * 1.0/self.n_sample_dropout
        else:
            out = self.drop(bo)
            output = self.out(out)

        return output

class TransformerSeqModel(nn.Module):
    def __init__(self, base_model, dropout=.1, n_out=1):
        super(TransformerSeqModel, self).__init__()

        self.base_model = base_model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0]
        bo = self.drop(o2)
        output = self.out(bo)

        return output

class TransformerWithMixout(nn.Module):
    def __init__(self, model, main_dropout_prob=0, mixout_prob=.7, dropout=.3, n_out=1):
        super(TransformerWithMixout, self).__init__()
        for i in range(model.config.num_hidden_layers):
            num = '{}'.format(i)
            for name, module in model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules.items():
                if name == 'dropout' and isinstance(module, nn.Dropout):
                    model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = nn.Dropout(main_dropout_prob)
                    #setattr(model, name, nn.Dropout(0))
                if name.split('.')[-1] == 'dense' and isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(module.in_features, module.out_features, 
                                           bias, target_state_dict['weight'], mixout_prob)
                    new_module.load_state_dict(target_state_dict)
                    #setattr(model, name, new_module)
                    model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = new_module
            
            #model._modules['drop'] = nn.Dropout(main_dropout_prob)
            
            #module = model._modules['out']
            #target_state_dict = module.state_dict()
            #bias = True if module.bias is not None else False
            #new_module = MixLinear(module.in_features, module.out_features, 
            #                       bias, target_state_dict['weight'], mixout_prob)
            #new_module.load_state_dict(target_state_dict)
                    
            #model._modules['out'] = new_module

        self.base_model = model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0]
        o2 = torch.mean(o2, dim=1)
        bo = self.drop(o2)
        output = self.out(bo)

        return output
