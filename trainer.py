import os
from tf_records_helper import RecordPrep
from utils import Lang
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()
"""
DataLoader is based on this article: 
    https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
"""
class DataLoader:

    def __init__(self):
        pass

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

    def make_dataset(self, path, batch_size=16):
        '''
        Makes  a Tensorflow dataset that is shuffled, batched and parsed.
        :param path: The path to a tf record file
        :param path: The size of our batch
        :return: a Dataset that shuffles and is padded
        '''
        # Read a tf record file. This makes a dataset of raw TFRecords
        dataset = tf.data.TFRecordDataset([path])
        # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
        dataset = dataset.map(RecordPrep.parse, num_parallel_calls=5)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=10000)
        # In order the pad the dataset, I had to use this hack to expand scalars to vectors.
        dataset = dataset.map(self.expand)
        # Batch the dataset so that we get batch_size examples in each batch.
        # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
        dataset = dataset.padded_batch(batch_size, padded_shapes={

            "length_1": 1,  # book_id is a scalar it doesn't need any padding, its always length one
            "length_2": 1,  # Likewise for the length of the sequence
            "vocab_ids": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
            "sense_ids": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
        })
        # Finally, we need to undo that hack from the expand function
        dataset = dataset.map(self.deflate)
        return dataset



    def prepare_dataset_iterators(self, path_to_records, batch_size=16):
        """

        :param path_to_records: path to TF Records directory
        :param batch_size:
        :return:
        """
        # Make a dataset from the train data
        train_ds = self.make_dataset(path_to_records, batch_size=batch_size)
        # make a dataset from the valdiation data
        #val_ds = make_dataset('./val.tfrecord', batch_size=batch_size)
        # Define an abstract iterator
        # Make an iterator object that has the shape and type of our datasets
        iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                   train_ds.output_shapes)

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()
        # These ops let us switch and reinitialize every time we finish an epoch
        training_init_op = iterator.make_initializer(train_ds)

        return next_element, training_init_op

class Encoder:

    def __init__(self, _n_layer, _hidden_size, _embedding_size):
        self.n_layers = _n_layer
        self.hidden_size = _hidden_size
        self.embedding_size = _embedding_size

