import os
import pickle
import gensim
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()


class Lang:

    def __init__(self, _vocab_path, _sense_map_path, _related_path, _antonyms_path,_embedding_size=200):

        self.index2word = pickle.load(open(_vocab_path, "rb"))
        self.word2index = dict()
        self.vocab_size = len(self.index2word)
        for key, value in self.index2word.items():
            self.word2index[value] = key

        self.sense_map = pickle.load(open(_sense_map_path, "rb"))
        self.related = pickle.load(open(_related_path, "rb"))
        self.antonyms = pickle.load(open(_antonyms_path, "rb"))
        self.embedding_size = _embedding_size
        self.gensim_word2vec = None

    def load_gensim_word2vec(self, word2vec_path):
        print("Loading pre-trained word embeddings...")
        self.gensim_word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        print("Word embeddings loaded.")

    def get_index2word(self):
        return self.index2word

    def create_embeddings(self):
        """
        Converts gensim word2vec dictionary to TensorFlow Tensor, mapping the indices
        to the ids from the index2word
        :param gensim_keyed: gensim object with loaded embeddings, queryable by word
        :return: TF Tensor [VOCAB_SIZE x EMBEDDING_SIZE]
        """

        w2v = np.zeros(shape=[self.vocab_size, self.embedding_size], dtype=np.float32)
        #w2v[0] = np.zeros(shape=[self.embedding_size])  # PAD
        words_found = 0
        words_missing = []
        for key, word in self.index2word.items():
            if key > 0:
                try:
                    w2v[key] = self.gensim_word2vec[word]
                    words_found += 1
                except KeyError as ke:
                    w2v[key] = np.random.normal(size=[1, self.embedding_size])
                    words_missing.append(word)

        print("words found: %d out of %d in vocab, difference is %d "
              % (words_found, len(self.index2word), (words_found - len(self.index2word))))

        return tf.contrib.eager.Variable(w2v)



if __name__ == "__main__":
    vocab_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/vocab/pickles/index2word.pickle"
    sense_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/sense_vocab/pickles/index2sense.pickle"
    related_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/related/relations.pickle"
    antonyms_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/related/relations_antonyms.pickle"
    embeddings_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/embeddings/GoogleNews-vectors-negative300.bin.gz"
    lang = Lang(vocab_path, sense_path, related_path, antonyms_path, _embedding_size=300)

    lang.load_gensim_word2vec(embeddings_path)
    print(lang.gensim_word2vec["dog"])
    #mbeddings = lang.create_embeddings()
    #print(tf.shape(embeddings))


