import gensim
import fasttext
import pandas as pd
import os

def generate_w2v(corpus,emb_path,tokenizer=None,min_count=2,size=200,epochs=5,window=3,sg=1,hs=1,negative=5,max_vocab_size=10000,sorted_vocab=1,seed=42):
    '''
    tokenizer supports NLTK tokenizer, spacy tokenizer, huggingface tokenizer
    NLTK -> use nltk.tokenize.WordPunctTokenizer().tokenize
    Spacy -> spacy_tokenizer('en_core_web_sm')
    Huggingface -> transformers.BertTokenizer.from_pretrained('bert-base-uncased').tokenize
    '''
    try:
        os.makedirs(emb_path)
    except OSError:
        pass

    if type(corpus[0]) == str:
        if tokenizer:
            corpus = [tokenizer(i) for i in corpus]
        else:
            corpus = [i.split() for i in corpus]

    model = gensim.models.word2vec.Word2Vec(min_count=min_count,size=size,window=window,sg=sg,hs=hs,negative=negative, \
                                    max_vocab_size=max_vocab_size,sorted_vocab=sorted_vocab,seed=seed)

    model.build_vocab(corpus)
    model.train(corpus,total_examples=model.corpus_count,epochs=epochs)

    model.wv.save_word2vec_format(os.path.join(emb_path,"w2v.txt"),binary=False)
    

def generate_fasttext(corpus,text_filepath,emb_path,cbow=False,min_count=2,minn=3, maxn=5, dim=200,epochs=5,lr=.1,neg=5,ws=5):

    try:
        os.makedirs(emb_path)
    except OSError:
        pass

    try:
        os.makedirs(text_filepath)
    except OSError:
        pass

    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    df = pd.DataFrame()
    df['text'] = corpus
    df.to_csv(os.path.join(text_filepath,'file.txt'),header=False,index=False)
    
    if cbow:
        model = fasttext.train_unsupervised(os.path.join(text_filepath,'file.txt'), os.path.join(emb_path,'ft'), "cbow", minCount=min_count, minn=minn, maxn=maxn, dim=dim, \
                                                epoch=epochs, lr=lr, ws=ws, neg=neg)

    else:
        model = fasttext.train_unsupervised(os.path.join(text_filepath,'file.txt'), os.path.join(emb_path,'ft'), minCount=min_count, minn=minn, maxn=maxn, dim=dim, \
                                                epoch=epochs, lr=lr, ws=ws, neg=neg)



