import xml

import MySQLdb
from collections import OrderedDict, Counter
import xml.etree.ElementTree as ET
import os, sys
import csv
import pickle

import numpy as np

"""
    Copy and reformat all the data to make sure that you are independent of ontonotes formatting in the future
"""

HOST = "Daniels-MacBook-Pro-2.local"
USER = "root"
PASSWORD = ""
DB_NAME = "ontoDB"


class DataLoader:
    def __init__(self, host, user, password, db_name, _corpus_dir=None):

        self.conn = MySQLdb.connect(host=host,
                                    user=user,
                                    passwd=password,
                                    db=db_name)
        # Word Dictionary
        self.word2index = {"PAD": 0}
        self.index2word = {0: "PAD"}
        self.word2count = Counter()
        self.n_words = 1
        # Sense Dictionary
        self.index2on_sense = {0:"PAD"}
        self.on_sense2index = {"PAD": 0}
        self.on_sense2count = Counter()
        self.n_senses = 1
        self.corpus_dir = _corpus_dir
        self.NO_TRACE_STRING = 4



    def ensure_dir(self):
        directory = self.corpus_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs("%s/%s/%s" % (directory, "ids", "csv_format"))
            os.makedirs("%s/%s/%s" % (directory, "ids", "pickles"))
            os.makedirs("%s/%s/%s" % (directory, "senses", "csv_format"))
            os.makedirs("%s/%s/%s" % (directory, "senses", "pickles"))
            os.makedirs("%s/%s/%s" % (directory, "vocab", "csv_format"))
            os.makedirs("%s/%s/%s" % (directory, "vocab", "pickles"))
            os.makedirs("%s/%s/%s" % (directory, "sense_vocab", "csv_format"))
            os.makedirs("%s/%s/%s" % (directory, "sense_vocab", "pickles"))





    """
        Database query methods below
    """
    def get_documents(self):
        """
        :return: list of all [[doc_id, subcorpus_id]] from ontonotes db
        """
        c = self.conn.cursor()
        c.execute("""SELECT id FROM document;""")
        doc_list = c.fetchall()
        #doc_list = [d[0] for d in doc_tuples]
        return doc_list

    def get_sentences(self, doc_id):
        """
        :param doc_id: match this doc and get all of it sentences (corpus)
        :return: list [[id , sentence_index, document_id,
        string (with some extra notation), no_trace_string (actual sentence content)]]
        """
        c = self.conn.cursor()
        c.execute("""SELECT * FROM sentence WHERE document_id = '%s';""" % doc_id)
        sentences = c.fetchall()
        return sentences

    def get_onsense(self, doc_id):
        """

        :param doc_id: match this doc and get all of sense tagged tokens from it
        :return: list [[id, lemma, pos, sense, word_index, tree_index, document_id]]
        """
        c = self.conn.cursor()
        c.execute("""SELECT id, lemma, pos, sense, word_index, tree_index, document_id 
                                           FROM on_sense 
                                           WHERE document_id = '%s' ;""" % doc_id)
        on_senses = c.fetchall()
        return on_senses

    """
        Ontonotes sense processing below
        
    """

    def add_sense(self, sense):
        """
        Helper for building the dictionary
        :param sense: string
        :return: int(id) for the sense
        """
        if sense not in self.on_sense2index:
            old_n_senses = self.n_senses
            self.on_sense2index[sense] = self.n_senses
            self.word2count[sense] = 1
            self.index2on_sense[self.n_senses] = sense
            self.n_senses += 1
            return old_n_senses
        else:
            self.on_sense2count[sense] += 1
            return self.on_sense2index[sense]

    """
        Dictionary/vocab helpers below
    """
    def add_word(self, word):
        """
        Helper for building the dictionary
        :param word: string
        :return: int(id) for the word
        """
        if word not in self.word2index:
            old_n_words = self.n_words
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            return old_n_words
        else:
            self.word2count[word] += 1
            return self.word2index[word]

    def add_sentence(self, sentence):
        """

        :param sentence:  a string representing a sentence
        :return: list of int(ids) corresponding to words in the sentence
        """
        ids = []
        for word in sentence.split(" "):
            ids.append(self.add_word(word.lower()))
        return ids

    def generate_tags(self, sentence_lens, doc_on_senses):
        """

        :param sentence_lens: list of sentence lengths
        :param doc_on_senses: a list of lists from ontoNotesDB of form
            [[id, lemma, pos, sense, word_index, tree_index, document_id]]
        :return: a list of lists of ints corresponding to ontoNotes senses
            0 is a padding/mask indicating that a word was not tagged with a sense
        """
        _LEMMA = 1
        _POS = 2
        _SENSE = 3
        _WORD_INDEX = 4
        _TREE_INDEX = 5
        doc_tags = []
        for sentence_len in sentence_lens:
            sentence_tags = [0 for i in range(sentence_len)]
            doc_tags.append(sentence_tags)

        for doc_on_sense in doc_on_senses:
            sense_string = "%s-%s-%s" % (doc_on_sense[_LEMMA], doc_on_sense[_POS], doc_on_sense[_SENSE])
            sense_id = self.add_sense(sense_string)
            doc_tags[doc_on_sense[_TREE_INDEX]][doc_on_sense[_WORD_INDEX]] = sense_id
        return doc_tags

    @staticmethod
    def export_doc(corpus_dir, doc_id, lists, ids_or_senses, write_pickle=True):
        """

        :param corpus_dir: path to a directory which stores converted files
        :param doc_id: an ID string of document being processed
        :param lists: list of lists [[sentence1_id1, ..., sentence1_idN], [sentence2_id1, ..., sentence2_idN]...]
        :param ids_or_senses: ids/senses, if id store words indices, if senses, corresponding ontonotes senses
        :param write_pickle: Bool, default true, flag decides whether to create pickle dumps
        :return: void
        """
        # By default export to csv
        corpus_file = open("%s/%s/csv_format/%s.csv" %(corpus_dir,ids_or_senses, doc_id.replace("/", '#')), "w")
        wr = csv.writer(corpus_file, dialect='excel')
        i = 0
        for id_list in lists:
            wr.writerow(id_list)

        if write_pickle:
            corpus_file = open("%s/%s/pickles/%s.pickle" %(corpus_dir,ids_or_senses, doc_id.replace("/", '#')), "w")
            pickle.dump(lists, corpus_file)

        print("Document %s was converted to %s and exported" % (ids_or_senses,doc_id))


    def build_corpus(self):
        self.ensure_dir()
        documents = self.get_documents()
        for document_id in documents:
            document_id = document_id[0]  # because it's a tuple with one element
            sentences = self.get_sentences(document_id)
            senses = self.get_onsense(document_id)
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = self.add_sentence(sentence[self.NO_TRACE_STRING])  # takes string, returns list of word ids
                doc_ids.append(sentence_ids)  # appends it to the document level list
                sentence_lens.append(len(sentence_ids))  # we need that later for sense tagging

            doc_senses = self.generate_tags(sentence_lens, senses)  # get senses in the form we want

            self.export_doc(self.corpus_dir, document_id, doc_ids, "ids")  # saves a list of list (sentences) of ids
            self.export_doc(self.corpus_dir, document_id, doc_senses, "senses") # saves a list of list (sentences) of senses

        export_vocab(self.index2word, self.n_words, self.word2count, self.corpus_dir)
        export_sense(self.index2on_sense, self.n_senses, self.on_sense2count, self.corpus_dir)


def export_vocab(index2word, n_words,word2count, dicts_dir, write_pickle=True, write_count=True):

    dict_file = open("%s/%s/csv_format/%s.csv" % (dicts_dir, "vocab", "index2word"), "w")
    wr = csv.writer(dict_file, dialect='excel')
    for key, value in index2word.items():
        wr.writerow((key, value))

    if write_count:
        counter_file = open("%s/%s/csv_format/%s.csv" % (dicts_dir, "vocab", "word2count"), "w")
        wr = csv.writer(counter_file, dialect='excel')
        wr.writerow(("UniqueCount", n_words))
        for key in word2count:
            wr.writerow((key, word2count[key]))

    if write_pickle:
        dict_file = open("%s/%s/pickles/%s.pickle" %(dicts_dir, "vocab", "index2word"), "w")
        pickle.dump(index2word, dict_file)
        if write_count:
            counter_file = open("%s/%s/pickles/%s.pickle" % (dicts_dir, "vocab", "word2count"), "w")
            pickle.dump(n_words, counter_file)
            pickle.dump(word2count, counter_file)


def export_sense(index2on_sense, n_senses, on_sense2count, dicts_dir, write_pickle=True):
    dict_file = open("%s/%s/csv_format/%s.csv" % (dicts_dir, "sense_vocab", "index2sense"), "w")
    wr = csv.writer(dict_file, dialect='excel')
    for key, value in index2on_sense.items():
        wr.writerow((key, value))

    counter_file = open("%s/%s/csv_format/%s.csv" % (dicts_dir, "sense_vocab", "sense2count"), "w")
    wr = csv.writer(counter_file, dialect='excel')
    wr.writerow(("UniqueCount", n_senses))
    for key in on_sense2count:
        wr.writerow((key, on_sense2count[key]))

    if write_pickle:
        dict_file = open("%s/%s/pickles/%s.pickle" % (dicts_dir, "sense_vocab", "index2sense"), "w")
        pickle.dump(index2on_sense, dict_file)
        counter_file = open("%s/%s/pickles/%s.pickle" % (dicts_dir, "sense_vocab", "sense2count"), "w")
        pickle.dump(n_senses, counter_file)
        pickle.dump(on_sense2count, counter_file)


def isvalid_wnsense(token):

    if token == "" or token == " " or token == "\n":
        return False
    try:
        int(token)
    except ValueError:
        return False
    return True


def parse_inventory_file(file_path, token, pos):
    """
    Extracts on_sense -> wordnet_sense relation from .xml inventory file
    :param file_path: absolute path to word-pos.xml sense inventory file
    :return: a list in form [[str(onto_sense), [int(wn_sense_1), ... ,int(wn_sense_n)]]
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except xml.etree.ElementTree.ParseError as err:
        print("parse error in file: %s", file_path)
        return []
    mappings = []
    for child in root.findall('sense'):
        onto_sense = child.get('n')
        if "." in onto_sense:
            for s in child.findall('mappings/wn'):
                    if s.text is not None:
                        #for a in s:
                        #    print(a)
                        if s.get('lemma') is not None:
                            if len(s.get('lemma')) > 0:
                                print("CHECK", s.get('lemma'), onto_sense)
                                wn_sense = []
                                for a in s.text.split(','):
                                    if isvalid_wnsense(a):
                                        wn_sense.append(int(a))
                                _onto_sense = s.get('lemma') + "."\
                                    + pos + "." + str(int(onto_sense.split(".")[0]) + int(onto_sense.split(".")[1]))
                                mappings.append((_onto_sense, wn_sense))
        else:
            for s in child.findall('mappings/wn'):
                if s.text is not None:
                    wn_sense = []
                    for a in s.text.split(','):
                        if isvalid_wnsense(a):
                            wn_sense.append(int(a))
                    _onto_sense = token + "." \
                        + pos + "." + onto_sense
                    mappings.append((_onto_sense, wn_sense))
    return mappings


def build_on_sense_wn_sense_map(sense_inventories_path):
    """
    Creates a map between on_senses and wordnet senses
    :param sense_inventories_path: absolute path to a directory with ontonotes sense inventory (.xml) files
    :return: OrderedDict on_to_wn[word-POS-on_sense] = [wn_senses]
    """
    _on_to_wn = OrderedDict()
    files = os.listdir(sense_inventories_path)
    for _file in files:
        try:
            pos = _file.replace(".xml", "").split("-")[1]
            token = _file.replace(".xml", "").split("-")[0]
        except IndexError as ie:
            print(ie, _file)
        on_wn_temp = parse_inventory_file("%s/%s" % (sense_inventories_path, _file), token, pos)
        for on_wn_temp_item in on_wn_temp:
            # on_to_wn[word-POS-on_sense] = [wn_senses]
            _on_to_wn[on_wn_temp_item[0]] = on_wn_temp_item[1]
            #_on_to_wn["%s-%s" % (_file.replace(".xml", ""), on_wn_temp_item[0])] = on_wn_temp_item[1]

    return _on_to_wn


def export_map(mappings, path, write_pickle=True):
    dict_file = open("%s/csv_format/%s.csv" % (path, "on2wn"), "w")
    wr = csv.writer(dict_file, dialect='excel')
    for key, value in mappings.items():
        wr.writerow((key, value))

    if write_pickle:
        dict_file = open("%s/pickles/%s.pickle" % (path, "on2wn"), "w")
        pickle.dump(mappings, dict_file)


if __name__ == '__main__':
    dataLoader = DataLoader(HOST, USER, PASSWORD, DB_NAME, _corpus_dir="corpus")
    dataLoader.build_corpus()
    #sense_inv = \
    #    "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/data/files/data/english/metadata/sense-inventories"
    #on_to_wn = build_on_sense_wn_sense_map(sense_inv)
    #export_map(on_to_wn, "corpus/sense_vocab")
    #m = \
    #    parse_inventory_file("/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/data/files/data/english/metadata/sense-inventories/get-v.xml", "get", "v")
    #print(m)