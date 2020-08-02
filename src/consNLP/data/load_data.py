import pandas as pd

def load_pandas_df(filepath, sep='\t', text_column=[], target_column=[]):
    df = pd.read_csv(filepath, sep=sep, quotechar='"')
    '''
    rename text column to words and target column to labels
    '''
    if len(text_column) == 1:
        df = df.rename({text_column[0]:'words'}, axis=1)

    if len(target_column) == 1:
        df = df.rename({target_column[0]:'labels'}, axis=1)
        
    return df

def load_custom_text_as_pd(filepath, sep='\t', header=True, text_column=[], target_column=[]):
    lines = open(filepath,'r').readlines()
    lines = [line.replace('\n','') for line in lines]
    
    if header:
        headers = lines[0].split(sep)
    else:
        headers = ["col_{}".format(i) for i in range(1,len(lines[0].split(sep))+1)]

    df = pd.DataFrame()

    for i, col in enumerate(headers):
        if header:
            df[col] = [line.split(sep)[i] for line in lines[1:]]
        else:
            df[col] = [line.split(sep)[i] for line in lines]

    if len(text_column) == 1:
        df = df.rename({text_column[0]:'words'}, axis=1)

    if len(target_column) == 1:
        df = df.rename({target_column[0]:'labels'}, axis=1)
        
    return df

class CoNLLData(object):
    def __init__(self, filepath, word_index=0, label_index=2, label_identifier='meta'):
        self.word_index = word_index
        self.label_index = label_index
        self.label_identifier = label_identifier

        self.words, self.start_list, self.end_list  = self.read_conll_format(filepath)
        self.labels = self.read_conll_format_labels(filepath)
        #assert len(self.words) == len(self.labels)
        self.sentence = ["sentence_{}".format(i+1) for i in range(len(self.words))]
        

    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                if line.split('\t')[0] == self.label_identifier and len(line.split('\t')) > 2:
                    probs = line.split("\t")[self.label_index]
                    post.append(probs)
                #print("post: ", post)
            elif len(post) > 0:
                posts.append(post[0])
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        start_list, end_list, starts, ends = [], [], [], []
        start, end = 0, 0
        for line in lines:
            if line:
                if len(line.split('\t')) <= 2:
                    start = end + 1
                    words = line.split("\t")[self.word_index]
                    end = start + len(words) 
                    # print("words: ", words)
                    post.append(words.lower())
                    starts.append(start)
                    ends.append(end)
            elif post:
                posts.append(post)
                start_list.append(starts)
                end_list.append(ends)
                post = []
                start, end = 0, 0
        # a list of lists of words/ labels
        return posts, start_list, end_list

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines

class CoNLLSeqData(object):
    def __init__(self, filepath, word_index=0, label_index=3):
        self.word_index = word_index
        self.label_index = label_index
        
        self.words, self.start_list, self.end_list  = self.read_conll_format(filepath)
        self.labels = self.read_conll_format_labels(filepath)

        #assert len(self.words) == len(self.labels)

        self.sentence = ["sentence_{}".format(i+1) for i in range(len(self.words))]
        

    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                probs = line.split("\t")[self.label_index]
                post.append(probs)
                #print("post: ", post)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        start_list, end_list, starts, ends = [], [], [], []
        start, end = 0, 0
        for line in lines:
            if line:
                start = end + 1
                words = line.split("\t")[self.word_index]
                end = start + len(words) 
                # print("words: ", words)
                post.append(words.lower())
                starts.append(start)
                ends.append(end)
            elif post:
                posts.append(post)
                start_list.append(starts)
                end_list.append(ends)
                post = []
                start, end = 0, 0
        # a list of lists of words/ labels
        return posts, start_list, end_list

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines
