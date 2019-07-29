import os
import pickle
import gensim
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()


class Lang:

    def __init__(self, _vocab_path, _sense_map_path, _related_path, word2vec_path,_embedding_size=200):

        self.index2word = pickle.load(open(_vocab_path, "rb"))
        self.word2index = dict()
        self.vocab_size = len(self.index2word)
        for key, value in self.index2word.items():
            self.word2index[value] = key

        self.sense_map = pickle.load(open(_sense_map_path, "rb"))
        self.related_ = pickle.load(open(_related_path, "rb"))
        self.embedding_size = _embedding_size


    def get_index2word(self):
        return self.index2word

    def create_embeddings(self, gensim_keyed):
        """
        Converts gensim word2vec dictionary to TensorFlow Tensor, mapping the indices
        to the ids from the index2word
        :param gensim_keyed: gensim object with loaded embeddings, queryable by word
        :return: TF Tensor [VOCAB_SIZE x EMBEDDING_SIZE]
        """

        w2v = np.zeros(self.vocab_size, self.embedding_size)
        w2v[0] = np.zeros(self.embedding_size)  # PAD
        words_found = 0
        words_missing = []
        for key, word in self.index2word.items():
            if key > 0:
                try:
                    w2v[key] = gensim_keyed[word]
                    words_found += 1
                except KeyError as ke:
                    w2v[key] = np.random.normal(size=[1, self.embedding_size])
                    words_missing.append(word)

        print("words found: %d out of %d in vocab, difference is %d "
              % (words_found, len(self.index2word), (words_found - len(self.index2word))))

        return tf.contrib.eager.Variable(w2v)


