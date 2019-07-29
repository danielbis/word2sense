import os
import pickle
import gensim
class Lang:

    def __init__(self, _vocab_path, _sense_map_path, _related_path):

        self.index2word = pickle.load(open(_vocab_path, "rb"))
        self.word2index = dict()
        for key, value in self.index2word.items():
            self.word2index[value] = key

        self.sense_map = pickle.load(open(_sense_map_path, "rb"))
        self.related_ = pickle.load(open(_related_path, "rb"))


    def get_index2word(self):
        return self.index2word

    def create_embeddings(self, embedding_matrix, target_lang, gensim_keyed):
        # 0-2 for pad, sos, eos
        embedding_matrix[0] = embedding_matrix[1] = embedding_matrix[2] = torch.ones(EMBEDDING_SIZE)
        words_found = 0
        words_missing = []
        for i in range(3, target_lang.n_words):
            try:
                embedding_matrix[i] = torch.from_numpy(gensim_keyed[target_lang.index2word[i]])
                words_found += 1
            except KeyError as ke:
                embedding_matrix[i] = torch.rand(200)
                words_missing.append(target_lang.index2word[i])

        print("words found: %d out of %d in vocab, difference is %d " \
              % (words_found, target_lang.n_words, (words_found - target_lang.n_words)))
        return torch.FloatTensor(embedding_matrix)


