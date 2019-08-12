import pickle
import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

VOCAB_PATH = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/ids/pickles"
SENSE_PATH = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/senses/pickles"
RECORDS_PATH = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/tf_records_corpus"


class RecordPrep:
    def __init__(self, _vocab_corpus_path, _sense_corpus_path, _tf_records_path):
        self.vocab_corpus_path = _vocab_corpus_path
        self.sense_corpus_path = _sense_corpus_path
        self.tf_records_path = _tf_records_path

    def sequence_to_tf_example(self, sequence_1, sequence_2):
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length_1 = len(sequence_1)  # list of word ids
        sequence_length_2 = len(sequence_2)  # list of sense ids
        if sequence_length_1 != sequence_length_2:
            raise Exception("Sequence lengths not equal: %d, %d" % (sequence_length_1, sequence_length_2))
        ex.context.feature["length_1"].int64_list.value.append(sequence_length_1)
        ex.context.feature["length_2"].int64_list.value.append(sequence_length_2)

        # Feature lists for the two sequential features of our example
        fl_tokens_1 = ex.feature_lists.feature_list["vocab_ids"]
        fl_tokens_2 = ex.feature_lists.feature_list["sense_ids"]

        for token in sequence_1:
            fl_tokens_1.feature.add().int64_list.value.append(token)

        for token in sequence_2:
            fl_tokens_2.feature.add().int64_list.value.append(token)

        return ex

    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go froma  serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length_1": tf.io.FixedLenFeature([], dtype=tf.int64),
            "length_2": tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "vocab_ids": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
            "sense_ids": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),

        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"vocab_ids": sequence_parsed["vocab_ids"], "length_1": context_parsed["length_1"],
                "sense_ids": sequence_parsed["sense_ids"], "length_2": context_parsed["length_2"]}

    def build_tfrecords(self):

        vocab_files = os.listdir(self.vocab_corpus_path)
        sense_files = os.listdir(self.sense_corpus_path)

        for _file in zip(vocab_files, sense_files):
            if _file[0].split('/')[-1] != _file[1].split('/')[-1]:
                raise Exception("Vocab and sense files do not match \n%s\n%s" %
                                (_file[0].split('/')[-1], _file[1].split('/')[-1]))
            else:
                vocab_document, sense_document = self.import_document_pickles(self.vocab_corpus_path + '/' + _file[0],
                                                                              self.sense_corpus_path + '/' + _file[1])
                self.serialize_doc(vocab_document, sense_document, _file[0])

    def check_if_tagged(self, seq):
        """
        checks if any word in the sequence is tagged with sense id.
        If so returns True, else False.
        :param seq: list of sense ids
        :return:
        """
        for w in seq:
            if w != 0:
                return True
        return False

    def import_document_pickles(self, vocab_corpus_path, sense_corpus_path):

        vocab_document = pickle.load(open(vocab_corpus_path, 'rb'))
        sense_document = pickle.load(open(sense_corpus_path, 'rb'))

        return vocab_document, sense_document

    def serialize_doc(self, vocab_list, sense_list, f_name):
        print(len(vocab_list))
        record_filename = self.tf_records_path + '/' + f_name + '.tfrecord'
        print("Serializing file %s into tfrecord, number of examples: %d" % (f_name, len(vocab_list)))

        with open(record_filename, 'w') as f:
            writer = tf.python_io.TFRecordWriter(f.name)
            for sentence in zip(vocab_list, sense_list):
                # filter out sentences without any words tagged with senses
                if self.check_if_tagged(sentence[1]):
                    example = self.sequence_to_tf_example(sentence[0], sentence[1])
                    writer.write(example.SerializeToString())

    def read_dataset(self):
        # Read a tf record file. This makes a dataset of raw TFRecords
        dataset = tf.data.TFRecordDataset(os.listdir(self.tf_records_path))
        # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
        dataset = dataset.map(self.parse, num_parallel_calls=5)
        # Shuffle the dataset
        # dataset = dataset.shuffle(buffer_size=10000)

        return dataset

if __name__ == '__main__':
    rprep = RecordPrep(VOCAB_PATH, SENSE_PATH, RECORDS_PATH)
    rprep.build_tfrecords()