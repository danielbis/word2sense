import sys, os
import csv
import pickle
import MySQLdb


HOST = "Daniels-MacBook-Pro-2.local"
USER = "root"
PASSWORD = ""
DB_NAME = "ontoDB"


class TestDataHandles:

    def __init__(self, host, user, password, db_name, corpus_path):

        self.conn = MySQLdb.connect(host=host,
                                    user=user,
                                    passwd=password,
                                    db=db_name)
        self.corpus_path = corpus_path
        self.vocab_path = corpus_path + "/vocab/csv_format/index2word.csv"
        self.documents_path = corpus_path + "/ids/csv_format/"
        self.word2index = {}

    def load_vocab(self):

        print("loading the vocabulary... ")
        dict_file = open(self.vocab_path, "r")

        vocab_reader = csv.reader(dict_file, dialect='excel')
        for row in vocab_reader:
            print(row)
            self.word2index[int(row[0])] = row[1]
        print("vocabulary loaded")

    def read_document(self, doc_id):

        print("reading a document %s ... " % doc_id)
        document = []
        doc_id = doc_id.replace("/", "#")
        with open(self.documents_path + doc_id + ".csv", 'r') as csvfile:
            doc_reader = csv.reader(csvfile, dialect='excel')
            for row in doc_reader:
                sentence = [self.word2index[int(i)] for i in row]
                document.append(sentence)
        print("document restored")

        return document

    def read_document_from_db(self, doc_id):

        c = self.conn.cursor()
        c.execute("""SELECT no_trace_string FROM sentence WHERE document_id = '%s';""" % doc_id)
        sentences = c.fetchall()
        return sentences

    def compare_docs(self, doc_id):

        restored = self.read_document(doc_id)
        from_db = self.read_document_from_db(doc_id)

        for i,s in enumerate(restored):
            print(" ".join(s))
            if " ".join(s) != from_db[i][0]:
                return False

        return True

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

if __name__ == '__main__':
    test_class = TestDataHandles(HOST, USER, PASSWORD, DB_NAME, corpus_path="corpus")
    test_class.load_vocab()
    works = test_class.compare_docs("bc/cctv/00/cctv_0001@all@cctv@bc@en@on")

    print(works)