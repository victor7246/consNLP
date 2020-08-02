import os
import torch
import numpy as np
import pickle
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import nltk
import transformers

from .preprocessing import tokenize_text

def flatten(elems):
    return [e for elem in elems for e in elem]

def _get_unique(elems):
    if type(elems[0]) == list:
        corpus = flatten(elems)
    else:
        corpus = elems
    elems, freqs = zip(*Counter(corpus).most_common())
    return list(elems)

def load_vocab_from_vectorizer(vectorizer,save_path=None):
    vocab = vectorizer.get_feature_names()
    word2index = {w:i+2 for i,w in enumerate(vocab)}

    if save_path:
        with open(save_path,'wb') as f:
            pickle.dump(word2index,f,-1)

    return word2index

def convert_categorical_label_to_int(labels,save_path=None):
    if type(labels[0]) == list:
        uniq_labels = _get_unique(flatten(labels))
    else:
        uniq_labels = _get_unique(labels)

    if os.path.exists(save_path):
        label_to_id = pickle.load(open(save_path,'rb'))

    else:
        if type(labels[0]) == list:
            label_to_id = {w:i+1 for i,w in enumerate(uniq_labels)}
        else:
            label_to_id = {w:i for i,w in enumerate(uniq_labels)}

    new_labels = []
    if type(labels[0]) == list:
        for i in labels:
            new_labels.append([label_to_id[j] for j in i])
    else:
        new_labels = [label_to_id[j] for j in labels]

    if save_path:
        with open(save_path,'wb') as f:
            pickle.dump(label_to_id,f,-1)

    return new_labels, label_to_id

def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip().split()
            if len(list(map(float, line[1:]))) > 1:
                if line[0].lower() not in word2index:
                    word2index[line[0].lower()] = i
                    embeddings.append(list(map(float, line[1:])))
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)

def process_data_for_transformers(text, bpetokenizer, tokenizer, max_len, target_text=None):
    #text = " " + " ".join(str(text).split())
    targets_start = 0
    targets_end = 0

    if target_text:
        #target_text = " " + " ".join(str(target_text).split())

        len_st = len(target_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(text) if e == target_text[1]):
            if " " + text[ind: ind+len_st] == target_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(text)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
    
    if bpetokenizer:
        tok_text = bpetokenizer.encode(text, max_length=max_len)
        input_ids_orig = tok_text.ids
        text_offsets = tok_text.offsets
    else:
        tok_text = tokenizer.encode(text, max_length=max_len)
        input_ids_orig = tokenizer.encode(text, max_length=max_len)
        text_offsets = [(0,0)] * len(tok_text)
    
    if target_text:
        target_idx = []
        for j, (offset1, offset2) in enumerate(text_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        targets_start = target_idx[0]
        targets_end = target_idx[-1]
        
    input_ids = [0] + input_ids_orig + [2]
    token_type_ids = [0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    text_offsets = [(0, 0)] + text_offsets + [(0, 0)]
    targets_start += 1
    targets_end += 1

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)
    
    if max_len > 0:
        return {
            'ids': input_ids[:max_len],
            'mask': mask[:max_len],
            'token_type_ids': token_type_ids[:max_len],
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_text': text,
            'orig_selected': target_text,
            'offsets': text_offsets,
            'label': None
        }
    else:
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_text': text,
            'orig_selected': target_text,
            'offsets': text_offsets,
            'label': None
        }

def process_data_for_transformers_conditional(text, label, all_labels, bpetokenizer, tokenizer, max_len, target_text=None):
    #text = " " + " ".join(str(text).split())
    targets_start = 0
    targets_end = 0

    if target_text:
        #target_text = " " + " ".join(str(target_text).split())

        len_st = len(target_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(text) if e == target_text[1]):
            if " " + text[ind: ind+len_st] == target_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(text)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
    
    if bpetokenizer:
        tok_text = bpetokenizer.encode(text, max_length=max_len)
        input_ids_orig = tok_text.ids
        text_offsets = tok_text.offsets
    else:
        tok_text = tokenizer.encode(text, max_length=max_len)
        input_ids_orig = tokenizer.encode(text, max_length=max_len)
        text_offsets = [(0,0)] * len(tok_text)
    
    all_label_ids = {}

    for _label in all_labels:
        _label_id = tokenizer.encode(_label)
        if len(_label_id) == 3:
            all_label_ids[_label] = _label_id[1]
        else:
            all_label_ids[_label] = tokenizer.unk_token_id

    if target_text:
        target_idx = []
        for j, (offset1, offset2) in enumerate(text_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        targets_start = target_idx[0]
        targets_end = target_idx[-1]

    input_ids = [0] + [all_label_ids[label]] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    text_offsets = [(0, 0)] * 3 + text_offsets + [(0, 0)]
    targets_start += 3
    targets_end += 3

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)
    
    if max_len > 0:
        return {
            'ids': input_ids[:max_len],
            'mask': mask[:max_len],
            'token_type_ids': token_type_ids[:max_len],
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_text': text,
            'orig_selected': target_text,
            'offsets': text_offsets,
            'label': label
        }
    else:
        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_text': text,
            'orig_selected': target_text,
            'offsets': text_offsets,
            'label': label
        }

class TransformerDataset:
    def __init__(self, text, bpetokenizer, tokenizer, MAX_LEN, target_label=None, tokenizer_target=False,
                sequence_target=False, target_text=None, conditional_label=None, conditional_all_labels=None):
        
        self.text = text
        self.target_label = target_label
        self.target_text = target_text
        self.conditional_label = conditional_label
        self.tokenizer = tokenizer
        self.bpetokenizer = bpetokenizer
        self.max_len = MAX_LEN
        self.conditional_all_labels = conditional_all_labels
        self.sequence_target = sequence_target
        self.tokenizer_target = tokenizer_target

        if target_label is None:
            self.target_label = [None]*len(self.text)

        if target_text is None:
            self.target_text = [None]*len(self.text)

        if conditional_label is None:
            self.conditional_label = [None]*len(self.text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        if type(text) == list:
            text = " ".join(text)

        conditional_label = self.conditional_label[item]
        target_text = self.target_text[item]
        target_label = self.target_label[item]

        if conditional_label:
            d = process_data_for_transformers_conditional(text, conditional_label, self.conditional_all_labels,
                                                 self.bpetokenizer, self.tokenizer, self.max_len, target_text)
        else:
            d = process_data_for_transformers(text, self.bpetokenizer, self.tokenizer, self.max_len, target_text) 

        if d["label"] is None:
            d["label"] = 0

        if d["orig_selected"] is None:
            d["orig_selected"] = ''

        if target_label:
            if self.sequence_target == True:
                if conditional_label:
                    target_label = [0]*3 + target_label
                    target_label = target_label + [0]*(self.max_len - len(target_label))
                else:
                    target_label = [0] + target_label
                    target_label = target_label + [0]*(self.max_len - len(target_label))

                target_label = target_label[:self.max_len]

            if self.tokenizer_target:

                target_dict = process_data_for_transformers(target_label, self.bpetokenizer, self.max_len)

                return {
                        "ids": torch.tensor(d['ids'], dtype=torch.long),
                        "mask": torch.tensor(d['mask'], dtype=torch.long),
                        "token_type_ids": torch.tensor(d['token_type_ids'], dtype=torch.long),
                        "targets": torch.tensor(target_dict['ids'], dtype=torch.long),
                        "targets_start": torch.tensor(d["targets_start"], dtype=torch.float),
                        "targets_end": torch.tensor(d["targets_end"], dtype=torch.float)
                    }
            else:

                return {
                        "ids": torch.tensor(d['ids'], dtype=torch.long),
                        "mask": torch.tensor(d['mask'], dtype=torch.long),
                        "token_type_ids": torch.tensor(d['token_type_ids'], dtype=torch.long),
                        "targets": torch.tensor(target_label, dtype=torch.float),
                        "targets_start": torch.tensor(d["targets_start"], dtype=torch.float),
                        "targets_end": torch.tensor(d["targets_end"], dtype=torch.float)
                    }
        else:
            return {
                        "ids": torch.tensor(d['ids'], dtype=torch.long),
                        "mask": torch.tensor(d['mask'], dtype=torch.long),
                        "token_type_ids": torch.tensor(d['token_type_ids'], dtype=torch.long),
                        "targets": torch.tensor(0, dtype=torch.float),
                        "targets_start": torch.tensor(d["targets_start"], dtype=torch.float),
                        "targets_end": torch.tensor(d["targets_end"], dtype=torch.float)
                    }    

class TransformerDatasetForMNLI:
    def __init__(self, text1, text2, tokenizer, MAX_LEN, target_label=None):
        
        self.text1 = text1
        self.text2 = text2
        self.target_label = target_label
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN

        if target_label is None:
            self.target_label = [None]*len(self.text)

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, item):
        text1 = self.text1[item]
        text2 = self.text2[item]

        if type(text1) == list:
            text1 = " ".join(text1)

        target_label = self.target_label[item]

        if not target_label:
            target_label = 0

        input_ids = self.tokenizer.encode(text1,text2,max_length=self.max_len,pad_to_max_length=True)

        return {
                    "ids": torch.tensor(input_ids, dtype=torch.long),
                    "mask": torch.tensor([1]*len(input_ids), dtype=torch.long),
                    "token_type_ids": torch.tensor([1]*len(input_ids), dtype=torch.long),
                    "targets": torch.tensor(target_label, dtype=torch.long)
                }

class Corpus(object):
    def __init__(self, dataset, tokenizer=nltk.tokenize.WordPunctTokenizer().tokenize):
        self.data = dataset
        try:
            self.data.words = [tokenizer.encode(i).ids for i in self.data.words]
            self.data.words  = [[tokenizer.id_to_token(j) for j in i] for i in self.data.words]
        except:
            self.data.words = [tokenizer(i) for i in self.data.words]

        #print (self.data.words)
        self.print_stats()

    def print_stats(self):
        data_length = [len(i) for i in self.data.words]
        print('[LOG]')
        print ("[LOG] Maximum word length of dataset {} and minimum length {}, median length {}, 90th percentile length {}".format(\
                        max(data_length),min(data_length),np.percentile(np.array(data_length),50), np.percentile(np.array(data_length),90)))

        try:
            data_length = [len(" ".join(i)) for i in self.data.words]
            print ("[LOG] Maximum character length of dataset {} and minimum length {}, median length {}, 90th percentile length {}".format(\
                        max(data_length),min(data_length),np.percentile(np.array(data_length),50), np.percentile(np.array(data_length),90)))
        except:
            pass

    def get_word_vocab(self):
        return _get_unique(self.data.words) #self.data.words + self.dev.words + self.test.words

    def get_char_vocab(self):
        return _get_unique([list(" ".join(i)) for i in self.data.words]) #self.data.words + self.dev.words + self.test.words

    def get_label_vocab(self):
        return _get_unique(self.data.labels) #self.data.labels + self.dev.labels + self.test.labels


class WordLevelData(object):
    def __init__(self, corpus, w2i_pkl_path, l2i_pkl_path, emb_npy_path, external_vocab=None, emb_path=None):

        self.corpus = corpus
        self.w2i_pkl_path = w2i_pkl_path
        self.l2i_pkl_path = l2i_pkl_path
        self.emb_npy_path = emb_npy_path
        self.emb_path = emb_path

        self.generate_dicts(external_vocab)

        self.print_stats()

    def generate_dicts(self,external_vocab=None):

        if self.emb_path :
            if external_vocab:
                self.word2index, self.word_emb = self.get_pretrain_embeddings(self.emb_path, external_vocab)
            else:
                self.word2index, self.word_emb = self.get_pretrain_embeddings(self.emb_path, self.corpus.get_word_vocab())

            np.save(self.emb_npy_path,self.word_emb)

            with open(self.w2i_pkl_path,'wb') as f:
                pickle.dump(self.word2index,f,-1)
        
        else:
            if os.path.exists(self.w2i_pkl_path):
                with open(self.w2i_pkl_path, 'rb') as fp:
                    self.word2index = pickle.load(fp)
                    self.word2index['+pad+'] = 0
                    self.word2index['+unk+'] = 1

            else:
                vocab = self.corpus.get_word_vocab()
                word2index = {'+pad+': 0, '+unk+': 1}
                for i, w in enumerate(vocab):
                    word2index[w] = i+2

                with open(self.w2i_pkl_path,'wb') as f:
                    pickle.dump(word2index,f,-1)

                self.word2index = word2index

        self.index2word = {i: w for w, i in self.word2index.items()}

        if os.path.exists(self.l2i_pkl_path):
            with open(self.l2i_pkl_path, 'rb') as fp:
                self.label2index = pickle.load(fp)

        else:
            if len(self.corpus.data.labels) == len(self.corpus.data.words):
                self.labels, self.label2index = convert_categorical_label_to_int(self.corpus.data.labels, self.l2i_pkl_path)

        self.index2label = {w:i for i,w in self.label2index.items()}


    def encode_words(self, corpus, max_len=None, pad_to_max_len=False):
        corpus.data.words = [self.encode(self.word2index, sample) for sample in corpus.data.words]
        if max_len and pad_to_max_len:
            corpus.data.words = [i + [self.word2index['+pad+']]*(max_len - len(i)) for i in corpus.data.words]

        return corpus

    def decode_words(self, corpus):
        corpus.data.words = [self.encode(self.index2word, sample) for sample in corpus.data.words]

        return corpus

    def encode_labels(self, corpus, max_len=None, pad_to_max_len=False):
        if type(corpus.data.labels[0]) == list:
            corpus.data.labels = [self.encode(self.label2index, sample) for sample in corpus.data.labels]
            if max_len and pad_to_max_len:
                corpus.data.labels = [i + [0]*(max_len - len(i)) for i in corpus.data.labels]
        else:
            corpus.data.labels = [self.encode(self.label2index, [sample])[0] for sample in corpus.data.labels]

        return corpus

    def decode_labels(self, corpus):
        if type(corpus.data.labels[0]) == list:
            corpus.data.labels = [self.encode(self.index2label, sample) for sample in corpus.data.labels]

        else:
            corpus.data.labels = [self.encode(self.index2label, [sample])[0] for sample in corpus.data.labels]

        return corpus

    def encode(self, elem2index, elems):
        return [elem2index[elem] for elem in elems]

    def print_stats(self):
        print('[LOG]')
        print("[LOG] Word vocab size: {}".format(len(self.word2index)))
        print("[LOG] labels vocab size: {}".format(len(self.label2index)))

    def get_pretrain_embeddings(self, filename, vocab):
        assert len(vocab) == len(set(vocab)), "The vocabulary contains repeated words"

        w2i, emb = read_text_embeddings(filename)
        word2index = {'+pad+': 0, '+unk+': 1}
        embeddings = np.zeros((len(vocab) + 2, emb.shape[1]))

        scale = np.sqrt(3.0 / emb.shape[1])
        embeddings[word2index['+unk+']] = np.random.uniform(-scale, scale, (1, emb.shape[1]))

        perfect_match = 0
        case_match = 0
        no_match = 0

        for i in range(len(vocab)):
            word = vocab[i]
            index = len(word2index)  # do not use i because word2index has predefined tokens

            word2index[word] = index
            if word in w2i:
                embeddings[index] = emb[w2i[word]]
                perfect_match += 1
            elif word.lower() in w2i:
                embeddings[index] = emb[w2i[word.lower()]]
                case_match += 1
            else:
                embeddings[index] = np.random.uniform(-scale, scale, (1, emb.shape[1]))
                no_match += 1
        print("[LOG] Word embedding stats -> Perfect match: {}; Case match: {}; No match: {}".format(perfect_match,
                                                                                                     case_match,
                                                                                                     no_match))
        return word2index, embeddings

class CharLevelData(object):
    def __init__(self, corpus, c2i_pkl_path, l2i_pkl_path, external_vocab=None):

        self.corpus = corpus
        self.c2i_pkl_path = c2i_pkl_path
        self.l2i_pkl_path = self.l2i_pkl_path

        self.generate_dicts(external_vocab)

        self.print_stats()

    def generate_dicts(self,external_vocab):

        if os.path.exists(self.c2i_pkl_path):
            with open(self.c2i_pkl_path, 'rb') as fp:
                self.char2index = pickle.load(fp)

        else:
            if external_vocab:
                char2index = self.get_vocab_idx(external_vocab)
            else:
                char2index = self.get_vocab_idx(self.corpus.get_char_vocab())

            with open(self.c2i_pkl_path,'wb') as f:
                pickle.dump(char2index,f,-1)

            self.char2index = char2index

        self.index2char = {i: w for w, i in self.char2index.items()}

        if os.path.exists(self.l2i_pkl_path):
            with open(self.l2i_pkl_path, 'rb') as fp:
                self.label2index = pickle.load(fp)

        else:
            if len(self.corpus.data.labels) == len(self.corpus.data.words):
                self.labels, self.label2index = convert_categorical_label_to_int(self.corpus.data.labels, self.l2i_pkl_path)

        self.index2label = {w:i for i,w in self.label2index.items()}


    def encode_chars(self, corpus, max_len=None, pad_to_max_len=False):
        corpus.data.words = [self.encode(self.char2index, " ".join(sample)) for sample in corpus.data.words]
        if max_len and pad_to_max_len:
            corpus.data.words = [i + [self.char2index['+pad+']]*(max_len - len(i)) for i in corpus.data.words]

        return corpus

    def decode_chars(self, corpus):
        corpus.data.words = [self.encode(self.index2char, sample) for sample in corpus.data.words]

        return corpus

    def encode_labels(self, corpus, max_len=None, pad_to_max_len=False):
        if type(corpus.data.labels[0]) == list:
            corpus.data.labels = [self.encode(self.label2index, sample) for sample in corpus.data.labels]
            if max_len and pad_to_max_len:
                corpus.data.labels = [i + [0]*(max_len - len(i)) for i in corpus.data.labels]
        else:
            corpus.data.labels = [self.encode(self.label2index, [sample])[0] for sample in corpus.data.labels]

        return corpus

    def decode_labels(self, corpus):
        if type(corpus.data.labels[0]) == list:
            corpus.data.labels = [self.encode(self.index2label, sample) for sample in corpus.data.labels]

        else:
            corpus.data.labels = [self.encode(self.index2label, [sample])[0] for sample in corpus.data.labels]

        return corpus
        
    def encode(self, elem2index, elems):
        return [elem2index[elem] for elem in elems]

    def print_stats(self):
        print('[LOG]')
        print("[LOG] char vocab size: {}".format(len(self.char2index)))
        print("[LOG] labels vocab size: {}".format(len(self.label2index)))

    def get_vocab_idx(self, vocab):

        char2index = {'+pad+': 0, '+unk+': 1}

        for i in range(len(vocab)):
            char = vocab[i]
            index = len(char2index)  # do not use i because word2index has predefined tokens

            if char not in char2index:
                char2index[char] = index

        return char2index