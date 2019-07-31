from nltk.corpus import wordnet as wn
import csv
import re
import pickle

"""
Questions: 
By how much is, on average, the coarser OntoNotes grouping increasing the number of synonyms, 
antonyms, and related words in general in comparison to WN?
    = sum(wn.lemma_names()) / sum.(onto_notes.lemma_names())
"""

####################### HELPERS #######################################################################################
CORPUS_DIR = "corpus"


def export_dicts_helper(mappings, dicts_dir,idx2token_path, token2idx_path, type, write_pickle=True):
    index2sense = dict()

    dict_file_1 = open("%s/csv_format/%s.csv" % (dicts_dir+type, token2idx_path), "w")
    dict_file_2 = open("%s/csv_format/%s.csv" % (dicts_dir+type, idx2token_path), "w")
    wr_1 = csv.writer(dict_file_1, dialect='excel')
    wr_2 = csv.writer(dict_file_2, dialect='excel')

    for key, value in mappings.items():
        wr_1.writerow((key, value))  # sense2index
        wr_2.writerow((value, key))  # index2sense
        index2sense[value] = key

    if write_pickle:
        dict_file_1 = open("%s/%s/pickles/%s.pickle" % (dicts_dir+type, token2idx_path), "w")
        dict_file_2 = open("%s/%s/pickles/%s.pickle" % (dicts_dir+type, idx2token_path), "w")

        pickle.dump(mappings, dict_file_1)
        pickle.dump(index2sense, dict_file_2)



def load_sense_mappings(path):
    mappings = {}
    print("Loading the OntoNotes --> WordNet sense mappings... ")
    dict_file = open(path, "r")
    converted = 0
    not_converted = 0

    vocab_reader = csv.reader(dict_file, dialect='excel')
    for row in vocab_reader:
        wn_senses = []
        for i in row[1][1:-1].split(','):  # get rid of quotation marks from csv files
            try:
                wn_senses.append(int(i.replace(" ", "")))  # saving the wn senses to a list
                converted +=1
            except ValueError as ve:  # inspecting the exceptions
                print("Can't convert", ve)
                print("ROW: ", row[0], row[1])
                not_converted +=1

        mappings[row[0]] = wn_senses  # create on_wn link
    print("###############################################################################")
    print("Done loading sense mappings, converted %s, could not convert %s." % (converted, not_converted))
    return mappings


def load_sense_mappings_pickle(path):
    mappings = pickle.load(open(path, "rb"))
    #print(mappings)
    for key, value in mappings.items():
        if len(value) == 0:
            del mappings[key]
    return mappings


def load_vocab(path):
    mappings = {}
    print("Loading the word --> index vocab mappings... ")
    dict_file = open(path, "r")
    vocab_reader = csv.reader(dict_file, dialect='excel')
    for row in vocab_reader:
        mappings[row[1]] = int(row[0])  # word = index

    return mappings

#######################################################################################################################


class OnWnMapper:
    def __init__(self, path_to_map, path_to_vocab, path_to_index2sense):
        """
        Important Note: During processing new words are added to on_senses and vocabulary,
        therefore they need to be exported again.
        :param path_to_map: ontonotes sense to wordnet sense mappings
        :param path_to_vocab: index2word mapping [.csv]
        :param path_to_index2sense: index2sense mapping [.pickle]
        """
        # sense mappings
        self.on2wn = load_sense_mappings_pickle(path_to_map)
        self.index2sense = load_sense_mappings_pickle(path_to_index2sense)
        self.sense2index = dict()
        for key, value in self.index2sense.items():
            self.sense2index[value.replace("-", ".")] = key

        self.n_onsenses = len(self.sense2index)
        # get rid of that, no longer needed, save memory
        del self.index2sense

        # Vocab
        self.word2index = load_vocab(path_to_vocab)
        self.n_words = len(self.word2index)
        # test
        print("Sense")
        print(self.sense2index["open.v.2"])
        print(self.on2wn["elaborate.v.1"])
        print(self.on2wn["elaborate.v.1"][0])
        print("Vocab:")
        print(self.word2index["elaborate"])

    def add_word(self, word):
        """
        Helper for building the dictionary
        :param word: string
        :return: int(id) for the word
        """
        old_n_words = self.n_words
        self.word2index[word] = self.n_words
        self.n_words += 1
        return old_n_words

    def add_onsense(self, on_sense):
        """
        Helper for building the dictionary
        :param word: string
        :return: int(id) for the word
        """
        old_onsenses = self.n_onsenses
        self.sense2index[on_sense] = self.n_onsenses
        self.n_onsenses += 1
        return old_onsenses



    def on2wn_format(self, on_sense):
        """
        creates  representation of words and their senses which
        can be used to query wordnet database (word.POS.sense_number)
        Example: home-n-5,[8] converts to home.n.8
        :param on_sense:
        :return: list of wordnet senses in proper format
        """
        wn_senses = []
        pattern = "\.\d+$"
        for i in self.on2wn[on_sense]:

            s = re.sub(pattern, ".%s" % str(i), on_sense)
            s = s.replace("-", ".")
            wn_senses.append(s)
        return wn_senses

    @staticmethod
    def _get_synonyms_helper(_wn_sense):
        """

        :param _wn_sense:
        :return: synonyms based on wordnet
        """
        #print(_wn_sense)
        try:
            _synset = wn.synset(_wn_sense)
        except:
            print(_wn_sense)
            return []
        return _synset.lemma_names()

    @staticmethod
    def _get_antonyms_helper(_wn_sense):
        _antonyms = []
        #print("antonyms for %s" % _wn_sense)
        try:
            for lemma in wn.synset(_wn_sense).lemmas():
                if lemma.antonyms():
                    _antonyms += [l.name() for l in lemma.antonyms()]
        except:
            print(_wn_sense)

        return _antonyms

    @staticmethod
    def _get_hypernyms_helper(_wn_sense):
        """
        :param _wn_sense:
        :return: list of hypernyms (specializations of a given word)
        """
        try:
            _synset = wn.synset(_wn_sense)
        except:
            print(_wn_sense)
            return []
        hypernyms = [lemma.name() for synset in _synset.hypernyms()
                     for lemma in synset.lemmas()]
        return hypernyms

    @staticmethod
    def _get_hyponyms_helper(_wn_sense):
        """
        :param _wn_sense:
        :return: list of hyponyms (generalization of a given word)
        """
        try:
            _synset = wn.synset(_wn_sense)
        except:
            print(_wn_sense)
            return []
        hyponyms = [lemma.name() for synset in _synset.hyponyms()
                    for lemma in synset.lemmas()]
        return hyponyms

    def get_synonyms(self, wn_senses):
        """

        :param wn_senses:
        :return: list of synsets from wordnet which can be considered synonymous based on ontonotes
        """

        synonyms = []
        for _wn_sense in wn_senses:
            synonyms += self._get_synonyms_helper(_wn_sense)
        return synonyms

    def get_antonyms(self, wn_senses):
        """
        :param wn_senses:
        :return:
        """
        antonyms = []
        for _wn_sense in wn_senses:
            antonyms += self._get_antonyms_helper(_wn_sense)

        return antonyms

    def get_hypernyms(self, wn_senses):
        """

        :param wn_senses:
        :return: a list of hypernyms for given on sense
        """

        hypernyms = []
        for _wn_sense in wn_senses:
            hypernyms += self._get_hypernyms_helper(_wn_sense)

        return hypernyms

    def get_hyponyms(self, wn_senses):
        """

        :param wn_senses:
        :return: a list of hyponyms for given on sense
        """

        hyponyms = []
        for _wn_sense in wn_senses:
            hyponyms += self._get_hyponyms_helper(_wn_sense)

        return hyponyms

    def get_related(self, on_sense):
        """

        :param on_sense:
        :return: lists: [ids of related words], [ids of antonyms]
        """
        wn_senses = self.on2wn_format(on_sense)
        _related = []
        _antonyms = []

        _related += self.get_synonyms(wn_senses)
        _related += self.get_hypernyms(wn_senses)
        _related += self.get_synonyms(wn_senses)

        _antonyms += self.get_antonyms(wn_senses)

        related = []
        antonyms = []

        for lemma in _related:
            if lemma in self.word2index:
                related.append(self.word2index[lemma])
            else:
                related.append(self.add_word(lemma))

        for lemma in _antonyms:
            if lemma in self.word2index:
                antonyms.append(self.word2index[lemma])
            else:
                antonyms.append(self.add_word(lemma))

        del _related  # free memory? It should go out of scope anyway...
        return related, antonyms

    def build_related(self):

        on2related = dict()
        on2antonym = dict()
        for key, value in self.on2wn.items():
            positively_related, antonyms = self.get_related(key)
            on2related[key] = positively_related
            on2antonym[key] = antonyms

        return on2related, on2antonym

    def build_export_related(self, path):
        MAX_RELATED = 128
        MAX_ANTONYMS = 8
        on2related = dict()
        on2antonyms = dict()
        with_antonyms = 0
        without_antonyms = 0
        max_related = 0
        max_antonyms = 0
        related_file = open(path + ".csv", "w")
        pickle_file = open(path + ".pickle", "w")

        antonyms_file = open(path + "_antonyms.csv", "w")
        antonyms_pickle = open(path + "_antonyms.pickle", "w")

        vocab_writer = csv.writer(related_file, dialect='excel')
        antonyms_writer = csv.writer(antonyms_file, dialect='excel')
        print("Exporting on_sense to related... ")
        for key, value in self.on2wn.items():
            _related, _antonyms = self.get_related(key)
            # csv
            if key in self.sense2index:
                vocab_writer.writerow([self.sense2index[key]] + [_related])
                antonyms_writer.writerow([self.sense2index[key]] + [_antonyms])
            else:
                vocab_writer.writerow([self.add_onsense(key)] + [_related])
                antonyms_writer.writerow([self.add_onsense(key)] + [_antonyms])

            if len(_antonyms) > 0:
                with_antonyms +=1
            else:
                without_antonyms += 1

            if len(_related) > max_related:
                max_related = len(_related)
            if len(_antonyms) > max_antonyms:
                max_antonyms = len(_antonyms)
            # to dump pickle, add paddings
            on2related[self.sense2index[key]] = _related + [0 for i in range(MAX_RELATED-len(_related))]

            on2antonyms[self.sense2index[key]] = _antonyms + [0 for i in range(MAX_ANTONYMS-len(_antonyms))]


        pickle.dump(on2related, pickle_file)
        pickle.dump(on2antonyms, antonyms_pickle)

        print("Exported on_sense to related mappings.")
        print("with antonyms: ", with_antonyms)
        print("without antonyms: ", without_antonyms)
        print("Max related length: ", max_related)
        print("Max antonyms length: ", max_antonyms)

    def export_updated_dicts(self, _cropus_dir):

        export_dicts_helper(mappings=self.sense2index, dicts_dir=CORPUS_DIR, type="sense_vocab",
                           idx2token_path="index2sense", token2idx_path="sense2index" )
        export_dicts_helper(mappings=self.word2index, dicts_dir=CORPUS_DIR, type="vocab",
                           idx2token_path="index2word", token2idx_path="word2index")




if __name__ == '__main__':
    PATH_VOCAB = \
        "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/vocab/csv_format/index2word.csv"
    PATH_ON2WN = \
        "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/sense_vocab/csv_format/on2wn.csv"
    PATH_ON2WN_PICKLE = \
        "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/sense_vocab/pickles/on2wn.pickle"
    PATH_RELATED = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/related/relations"
    PATH_INDEX2SENSE = \
        "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/sense_vocab/pickles/index2sense.pickle"
    mapperObject = OnWnMapper(PATH_ON2WN_PICKLE, PATH_VOCAB, PATH_INDEX2SENSE)
    mapperObject.build_export_related(PATH_RELATED)


