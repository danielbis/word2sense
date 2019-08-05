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

    def deflate(self,x):
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
        })
        # Finally, we need to undo that hack from the expand function
        dataset = dataset.map(self.deflate)
        dataset = dataset.prefetch(2)
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


