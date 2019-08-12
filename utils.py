import os
import pickle
import gensim
import numpy as np
import csv
from tf_records_helper import RecordPrep
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
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

    def create_temp_embeddings(self):
        w2v = np.zeros(shape=[self.vocab_size, self.embedding_size], dtype=np.float32)
        return tf.contrib.eager.Variable(w2v)

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


"""
    EvalDataLoader is based on this article: 
    https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
"""
class EvalDataLoader:
    def expand(self, x):
        """
        Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar  to a vector of length 1
        :param x:
        :return:
        """
        x['idx'] = tf.expand_dims(tf.convert_to_tensor(x['idx']), 0)
        x['length1'] = tf.expand_dims(tf.convert_to_tensor(x['length1']), 0)
        x['length2'] = tf.expand_dims(tf.convert_to_tensor(x['length2']), 0)
        x['word1idx'] = tf.expand_dims(tf.convert_to_tensor(x['word1idx']), 0)
        x['word2idx'] = tf.expand_dims(tf.convert_to_tensor(x['word2idx']), 0)
        x['word1'] = tf.expand_dims(tf.convert_to_tensor(x['word1']), 0)
        x['word2'] = tf.expand_dims(tf.convert_to_tensor(x['word2']), 0)
        x['avg_rating'] = tf.expand_dims(tf.convert_to_tensor(x['avg_rating']), 0)

        return x

    def deflate(self, x):
        """
        Undo Hack. We undo the expansion we did in expand
        """
        x['idx'] = tf.squeeze(tf.convert_to_tensor(x['idx']))
        x['length1'] = tf.squeeze(tf.convert_to_tensor(x['length1']))
        x['length2'] = tf.squeeze(tf.convert_to_tensor(x['length2']))
        x['word1idx'] = tf.squeeze(tf.convert_to_tensor(x['word1idx']))
        x['word2idx'] = tf.squeeze(tf.convert_to_tensor(x['word2idx']))
        x['word1'] = tf.squeeze(tf.convert_to_tensor(x['word1']))
        x['word2'] = tf.squeeze(tf.convert_to_tensor(x['word2']))
        x['avg_rating'] = tf.squeeze(tf.convert_to_tensor(x['avg_rating']))

        return x

    @staticmethod
    def parse(ex):
        """
        Explain to TF how to go from a  serialized example back to tensors
        :param ex: Entry from tf record
        :return:
        """
        context_features = {
            "idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # example id from scws
            "length1": tf.io.FixedLenFeature([], dtype=tf.int64),  # length of the fist sentence
            "length2": tf.io.FixedLenFeature([], dtype=tf.int64),  # length of the second sentence
            "word1idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # word 1 position in sentence
            "word2idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # word 2 position in sentence
            "word1": tf.io.FixedLenFeature([], dtype=tf.int64),  # word 1 id
            "word2": tf.io.FixedLenFeature([], dtype=tf.int64),  # word 2 id
            "avg_rating": tf.io.FixedLenFeature([], dtype=tf.float32)  # average similarity rating from scws
        }
        sequence_features = {
            "sentence1": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "sentence2": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),

        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"idx": context_parsed["idx"], "length1": context_parsed["length1"],
                "length2": context_parsed["length2"], "word1idx": context_parsed["word1idx"],
                "word2idx": context_parsed["word2idx"], "word1": context_parsed["word1"],
                "word2": context_parsed["word2"], "avg_rating": context_parsed["avg_rating"],
                "sentence1": sequence_parsed["sentence1"], "sentence2": sequence_parsed["sentence2"]}

    def make_dataset(self, file_list, batch_size=16):
        '''
        Makes  a Tensorflow dataset that is shuffled, batched and parsed.
        :param file_list: The path to a tf record file
        :param path: The size of our batch
        :return: a Dataset that shuffles and is padded
        '''
        # Read a tf record file. This makes a dataset of raw TFRecords
        dataset = tf.data.TFRecordDataset(file_list)
        # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
        dataset = dataset.map(self.parse, num_parallel_calls=2)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=10000)
        # In order the pad the dataset, I had to use this hack to expand scalars to vectors.
        dataset = dataset.map(self.expand)
        # Batch the dataset so that we get batch_size examples in each batch.
        # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
        dataset = dataset.padded_batch(batch_size, padded_shapes={

            "idx": 1,  # book_id is a scalar it doesn't need any padding, its always length one
            "length1": 1,  # Likewise for the length of the sequence
            "length2": 1,
            "word1idx": 1,
            "word2idx": 1,
            "word1": 1,
            "word2": 1,
            "avg_rating": 1,
            "sentence1": tf.TensorShape([None]),  # but the seqeunce is variable length, we pass that information to TF
            "sentence2": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
        }, drop_remainder=True)
        # Finally, we need to undo that hack from the expand function
        dataset = dataset.map(self.deflate)
        dataset = dataset.prefetch(4)
        return dataset


##############################################################################################################
"""
DataLoader is based on this article: 
    https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
"""

class DataLoader:
    def expand(self, x):
        """
        Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar  to a vector of length 1
        :param x:
        :return:
        """
        x['length_1'] = tf.expand_dims(tf.convert_to_tensor(x['length_1']), 0)
        x['length_2'] = tf.expand_dims(tf.convert_to_tensor(x['length_2']), 0)
        return x

    def deflate(self, x):
        """
        Undo Hack. We undo the expansion we did in expand
        """
        x['length_1'] = tf.squeeze(x['length_1'])
        x['length_2'] = tf.squeeze(x['length_2'])
        return x

    def make_dataset(self, file_list, batch_size=16):
        '''
        Makes  a Tensorflow dataset that is shuffled, batched and parsed.
        :param file_list: The path to a tf record file
        :param path: The size of our batch
        :return: a Dataset that shuffles and is padded
        '''
        # Read a tf record file. This makes a dataset of raw TFRecords
        dataset = tf.data.TFRecordDataset(file_list)
        # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
        dataset = dataset.map(RecordPrep.parse, num_parallel_calls=2)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=10000)
        # In order the pad the dataset, I had to use this hack to expand scalars to vectors.
        dataset = dataset.map(self.expand)
        # Batch the dataset so that we get batch_size examples in each batch.
        # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
        dataset = dataset.padded_batch(batch_size, padded_shapes={

            "length_1": 1,  # book_id is a scalar it doesn't need any padding, its always length one
            "length_2": 1,  # Likewise for the length of the sequence
            "vocab_ids": tf.TensorShape([None]),  # but the seqeunce is variable length, we pass that information to TF
            "sense_ids": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
        }, drop_remainder=True)
        # Finally, we need to undo that hack from the expand function
        dataset = dataset.map(self.deflate)
        dataset = dataset.prefetch(4)
        return dataset



def prepare_dataset_iterators(self, file_list, batch_size=4):
    """

    :param path_to_records: path to TF Records directory
    :param batch_size:
    :return:
    """
    # Make a dataset from the train data
    train_ds = self.make_dataset(file_list, batch_size=batch_size)
    # make a dataset from the valdiation data
    #val_ds = make_dataset('./val.tfrecord', batch_size=batch_size)
    # Define an abstract iterator
    # Make an iterator object that has the shape and type of our datasets
    #iterator = tf.data.Iterator.from_structure(train_ds.output_types,
    #                                           train_ds.output_shapes)

    # This is an op that gets the next element from the iterator
    #next_element = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    #training_init_op = iterator.make_initializer(train_ds)

    return train_ds


def load_related(path2related, max_size=128):
    """
      Load and pad the related words mapping
      :param path2realted
      :param max_size (max number of related)
      :return: np.array, shape = [total_senses x max_size]
    """
    related = pickle.load(open(path2related, "rb"))
    print("MAX: ", max(related.keys()))
    related_matrix = np.zeros(shape=[max(related.keys())+1, max_size], dtype=np.int32)
    print(related[34])

    for key, value in related.items():
        #print(key, value)
        diff = range(len(value) - max_size)
        value = value + [0 for i in diff]
        related_matrix[key] = value

    return related_matrix


def export_dicts_helper(mappings, dicts_dir,idx2token_path, token2idx_path, type, write_pickle=True):
    index2sense = dict()

    dict_file_1 = open("%s/%s/csv_format/%s.csv" % (dicts_dir,type, token2idx_path), "w")
    dict_file_2 = open("%s/%s/csv_format/%s.csv" % (dicts_dir,type, idx2token_path), "w")
    wr_1 = csv.writer(dict_file_1, dialect='excel')
    wr_2 = csv.writer(dict_file_2, dialect='excel')

    for key, value in mappings.items():
        wr_1.writerow((key, value))  # sense2index
        wr_2.writerow((value, key))  # index2sense
        index2sense[value] = key

    print("I2S: ", len(index2sense))
    print("MAX KEY: ", max(index2sense.keys()))
    if write_pickle:
        dict_file_1 = open("%s/%s/pickles/%s.pickle" % (dicts_dir,type, token2idx_path), "w")
        dict_file_2 = open("%s/%s/pickles/%s.pickle" % (dicts_dir,type, idx2token_path), "w")

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
    """
    for key, value in mappings.items():
        if len(value) == 0:
            del mappings[key]
    """
    return mappings


def load_vocab(path):
    mappings = {}
    print("Loading the word --> index vocab mappings... ")
    dict_file = open(path, "r")
    vocab_reader = csv.reader(dict_file, dialect='excel')
    for row in vocab_reader:
        mappings[row[1]] = int(row[0])  # word = index

    return mappings

if __name__ == "__main__":
    vocab_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/vocab/pickles/index2word.pickle"
    sense_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/sense_vocab/pickles/index2sense.pickle"
    related_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/related/relations.pickle"
    antonyms_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/related/relations_antonyms.pickle"
    embeddings_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/embeddings/GoogleNews-vectors-negative300.bin.gz"
    #lang = Lang(vocab_path, sense_path, related_path, antonyms_path, _embedding_size=300)

    #lang.load_gensim_word2vec(embeddings_path)
    #print(lang.gensim_word2vec["dog"])
    #mbeddings = lang.create_embeddings()
    #print(tf.shape(embeddings))
    related = load_related(related_path)
    print(related[34])
    print(related[0])


