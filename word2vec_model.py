import numpy as np
import pandas as pd
import csv


UNKNOWN_CHAR = u"UNKNOWN"


def text_to_vec(input_sentence, lookup_dictionary, embeded_matrix, post_length):
    '''
    turn a text input to a fixed length vector according the word2vector
    dictionary and embed matrix
    @param input_sentence: a text string
    @param lookup_dictionary: word2vec model
    @param embeded_matrix: embed matrix to look up
    @param post_length: the required length of a post
    @return: a vector of post_length siz, padding with 0
    if length of post is less than post_length
    '''
    character_index = []
    # index value for charactor not recorded in the lookup_dictionary
    unknown_indx = lookup_dictionary[UNKNOWN_CHAR]
    character_index = [lookup_dictionary.get(character, unknown_indx)
                       for character in input_sentence]
    vec = embeded_matrix[character_index, :]
    if vec.shape[0] <= post_length:
        vec = np.lib.pad(
            vec,
            [(0, post_length - vec.shape[0]), (0, 0)],
            'constant',
            constant_values=(0)
        )
    return vec[0: post_length, :]


class MapWord2Vec(object):

    def __init__(self, path):
        '''
        Create a Word2vec model from an input word2vec model path
        word_index_table is the table to look for the index of each word
        embeded_matrix is the table to look for vector for each word
        @param path: model path
        '''
        word2vec_df = pd.read_csv(
            path,
            header=None, delimiter=",",
            quoting=csv.QUOTE_NONE,
            encoding='utf-8'
        )
        words_indexs = word2vec_df
        self.word_index_table = {}
        for word in range(len(words_indexs[0])):
            if word == 0:
                self.word_index_table[UNKNOWN_CHAR] = word
            else:
                self.word_index_table[words_indexs[0][word]] = word
        self.embeded_matrix = word2vec_df.iloc[:, 1:].as_matrix()