import gensim
from tqdm import tqdm
import numpy as np

import tensorflow_hub as hub
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            try:
                line = line.strip().split()
                if len(list(map(float, line[1:]))) > 1:
                    if line[0].lower() not in word2index:
                        word2index[line[0].lower()] = i
                        embeddings.append(list(map(float, line[1:])))
            except ValueError:
                pass
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)

def generate_embeddings_from_words(text,emb_file_path,tokenizer=None):
    word2index, embeddings = read_text_embeddings(emb_file_path)

    mapping = {}

    for i, key in enumerate(list(word2index.keys())):
        mapping[key] = embeddings[i]

    if type(str) == str:
        if tokenizer:
            text = tokenizer(text)
        else:
            text = i.split()

    vec = np.zeros(embeddings.shape[1])
    count = 0
    for i in text:
        if i in mapping:
            vec += mapping[i]
            count += 0

    if count > 0:
        vec = vec/count

    return vec

def w2v_embedding_corpus(corpus,emb_file_path,tokenizer=None):
    word2index, embeddings = read_text_embeddings(emb_file_path)

    scale = np.sqrt(3.0 / embeddings.shape[1])

    vector = np.random.uniform(-scale, scale, embeddings.shape)

    for i, text in enumerate(corpus):
        vector[i] = generate_embeddings_from_words(text,emb_file_path,tokenizer)

    return vector

def generate_doc2vec(corpus,emb_path,tokenizer=None,min_count=2,size=200,epochs=5,window=3,hs=1,negative=5,max_vocab_size=10000,seed=42):
    #if type(corpus[0]) == list:
    #        corpus = [" ".join(i) for i in corpus]

    if type(corpus[0]) == str:
        if tokenizer:
            corpus = [tokenizer(i) for i in corpus]
        else:
            corpus = [i.split() for i in corpus]

    documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = gensim.models.doc2vec.Doc2Vec(min_count=min_count,vector_size=size,window=window,hs=hs,negative=negative, \
                                    max_vocab_size=max_vocab_size,seed=seed)
    model.build_vocab(documents)
    model.train(documents,total_examples = model.corpus_count,epochs=epochs)

    try:
        os.makedirs(emb_path)
    except OSError:
        pass

    vector = np.array([model.docvecs[i] for i in range(len(corpus))])

    model.save(os.path.join(emb_path,"d2v.bin"))

    return vector

def universal_sentence_encoding(corpus,batch_size=32):

    if type(corpus[0]) == list:
        corpus = [" ".join(i) for i in corpus]

    g = tf.Graph()
    with g.as_default():
      text_input = tf.placeholder(dtype=tf.string, shape=[None])
      embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
      embedded_text = embed(text_input)
      init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    session = tf.Session(graph=g)
    session.run(init_op)

    vector = []
    pbar = tqdm(total=len(corpus)//batch_size)

    ind = 0
    while ind*batch_size < len(corpus):
        vector.append(session.run(embedded_text, feed_dict={text_input: corpus[ind*batch_size: (ind + 1)*batch_size]}))
        ind += 1
        pbar.update(1)

    pbar.close()

    vector = np.vstack(vector)

    return vector
