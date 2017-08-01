import pandas as pd 
import csv
import re
import numpy as np 
from word2vec_model import MapWord2Vec
import codecs 
import random 


class SentimentDataSet(object):
    
    TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)
    model = MapWord2Vec("word2vector_model.csv")
    vocabulary_ = model.word_index_table
    vector_matrix = model.embeded_matrix

    def __init__(self, input_path, max_document_length):
        self.input_path = input_path
        self.max_document_length = max_document_length
    def group_words(self, text):
        regex = []

        # Match a whole word:
        regex += [ur'\w+']

        # Match a single CJK character:
        regex += [ur'[\u4e00-\ufaff]']

        # Match one of anything else, except for spaces:
        regex += [ur'[^\s]']

        regex = "|".join(regex)
        r = re.compile(regex)
        return r.findall(text)

    def tokenizer(self, iterator):
        for value in iterator:
            yield self.TOKENIZER_RE.findall(" ".join(self.group_words(value)))

    def import_data(self, train_test_ratio=0.8):
        # Get input data path, the data is in form of original_review{delimiter:|}[1|0]
        input_file = codecs.open(self.input_path, encoding='utf-8')
        # Get line by line data
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        data_size = 0
        for line in input_file:
            data_size += 1
            parts = line.split(u"|")
            if random.uniform(0, 1) < train_test_ratio:
                if "1" in parts[1]:
                    train_y.append([1, 0])  # python will convert \n to os.linesep
                    train_x.append(parts[0])
                elif "0" in parts[1]:
                    train_y.append([0, 1])  # python will convert \n to os.linesep
                    train_x.append(parts[0])

            else:
                if "1" in parts[1]:
                    test_y.append([1, 0])  # python will convert \n to os.linesep
                    test_x.append(parts[0])
                elif "0" in parts[1]:
                    test_y.append([0, 1])  # python will convert \n to os.linesep
                    test_x.append(parts[0])
        input_file.close()
    
        train_vec_x = np.array(list(self.transform(train_x, self.max_document_length)))
        test_vec_x = np.array(list(self.transform(test_x, self.max_document_length)))
        return (train_vec_x, train_y), (test_vec_x, test_y)

    def transform(self, raw_documents, unused_y=None):
        for tokens in self.tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(list(tokens)):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token, 0)
            yield word_ids
