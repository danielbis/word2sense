import os
import pickle
import tensorflow as tf
from utils import export_dicts_helper, load_vocab


class EvalDataLoader:
    def __init__(self, path_to_vocab_pickle, path_to_data):
        """
                This class is used to load evaluation data and export data to TfRecords.
                Important Note: Modifies vocab files!
                During processing new words are added to the vocabulary,
                therefore they need to be exported again.
                :param path_to_vocab_pickle: string
                :param path_to_data path: to a txt file with scws evaluation data

        """

        self.word2index = pickle.load(open(path_to_vocab_pickle, "rb"))
        self.n_words = len(self.word2index)
        self.examples = self.parse_eval_data(path_to_data)

        print(len(self.word2index))

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

    @staticmethod
    def parse_eval_data(path_to_data):
        """

        :param path_to_data: path to a txt file with scws evaluation data
        :return: list of dicts (example) corresponding to one line of the dataset.
            example['idx'] = int(id)
            example['word1'] = string
            example['word2'] = string
            example['pos1'] = string
            example['pos2'] = string
            example['avg_rating'] = float, average human rating of relation
            example["sentence1"] = list of tokens from parsed first sentence without <b>,</b> tokens
            example["sentence2"] = list of tokens from parsed second sentence without <b>,</b> tokens
            example["word1idx"] = word1 index in the example["sentence1"] after parsing
            example["word2idx"] = word2 index in the example["sentence2"] after parsing

        """
        _file = open(path_to_data, "r")
        examples = []
        for line in _file:
            example = dict()

            _split = line.lower().split("\t")
            example['idx'] = int(_split[0])
            example['word1'] = _split[1]
            example['word2'] = _split[3]
            example['pos1'] = _split[2]
            example['pos2'] = _split[4]
            example['avg_rating'] = float(_split[7])

            seq1 = _split[5].split(" ")
            seq2 = _split[6].split(" ")

            for i, w in enumerate(seq1):
                if w == example['word1'] and seq1[i-1] == "<b>":
                    # position in the sentence
                    example["word1idx"] = i - 1  # we are about to remove <b> and </b>
                    break
            seq1.remove('<b>')
            seq1.remove('</b>')

            for i, w in enumerate(seq2):
                if w == example['word2'] and seq2[i-1] == "<b>":
                    # position in the sentence
                    example["word2idx"] = i - 1  # we are about to remove <b> and </b>
                    break
            seq2.remove('<b>')
            seq2.remove('</b>')

            example["sentence1"] = seq1
            example["sentence2"] = seq2

            examples.append(example)

        return examples

    def sequence_to_tf_example(self, example):
        """

        :param example: dict
            example['idx'] = int(id)
            example['word1'] = string
            example['word2'] = string
            example['pos1'] = string
            example['pos2'] = string
            example['avg_rating'] = float, average human rating of relation
            example["sentence1"] = list of tokens from parsed first sentence without <b>,</b> tokens
            example["sentence2"] = list of tokens from parsed second sentence without <b>,</b> tokens
            example["word1idx"] = word1 index in the example["sentence1"] after parsing
            example["word2idx"] = word2 index in the example["sentence2"] after parsing
        :return: TensorFlow Record serialized representation of example above.
        Without pos1 and pos2
        example[length1] and example[length2] contains corresponding sequence lengths
        """
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        not_found = 0
        sequence_1 = []
        sequence_2 = []
        for word in example['sentence1']:
            try:
                sequence_1.append(self.word2index[word])
            except KeyError as ke:
                sequence_1.append(self.add_word(word))

        for word in example['sentence2']:
            try:
                sequence_2.append(self.word2index[word])
            except KeyError as ke:
                sequence_2.append(self.add_word(word))
        try:
            ex.context.feature["word1"].int64_list.value.append(self.word2index[example['word1']])
        except KeyError:
            ex.context.feature["word1"].int64_list.value.append(self.add_word(example['word1']))
        try:
            ex.context.feature["word2"].int64_list.value.append(self.word2index[example['word2']])
        except KeyError:
            ex.context.feature["word2"].int64_list.value.append(self.add_word(example['word2']))

        sequence_length_1 = len(sequence_1)  # list of word ids
        sequence_length_2 = len(sequence_2)  # list of sense ids
        # example id
        ex.context.feature["idx"].int64_list.value.append(example['idx'])

        ex.context.feature["length1"].int64_list.value.append(sequence_length_1)
        ex.context.feature["length2"].int64_list.value.append(sequence_length_2)
        # position in the sentence
        ex.context.feature["word1idx"].int64_list.value.append(example['word1idx'])
        ex.context.feature["word2idx"].int64_list.value.append(example['word2idx'])

        ex.context.feature["avg_rating"].float_list.value.append(example['avg_rating'])

        # Feature lists for the two sequential features of our example
        fl_tokens_1 = ex.feature_lists.feature_list["sentence1"]
        fl_tokens_2 = ex.feature_lists.feature_list["sentence2"]

        for token in sequence_1:
            fl_tokens_1.feature.add().int64_list.value.append(token)

        for token in sequence_2:
            fl_tokens_2.feature.add().int64_list.value.append(token)

        return ex, not_found

    def serialize_examples(self):
        """
        Exports the dataset to tf records.
        :return:
        """
        valid_record_filename = 'scws_records/scvs_valid.tfrecord'
        test_record_filename = 'scws_records/scvs_test.tfrecord'

        test_record = open(test_record_filename, 'w')
        valid_record = open(valid_record_filename, 'w')
        writer_test = tf.python_io.TFRecordWriter(test_record.name)
        writer_valid = tf.python_io.TFRecordWriter(valid_record.name)
        not_found = 0
        for i, ex in enumerate(self.examples):
                # print(sentence[0], sentence[1])
            example, nf = self.sequence_to_tf_example(ex)
            not_found += nf
            if i % 5 == 0:
                writer_valid.write(example.SerializeToString())
            else:
                writer_test.write(example.SerializeToString())

        print("Saved eval data to TFRecords. Not found: ", not_found)


if __name__ == "__main__":
    vocab_pickle = \
        "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/corpus/vocab/pickles/word2index.pickle"
    scws_path = "/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/SCWS/ratings.txt"
    eval_data_loader = EvalDataLoader(vocab_pickle, scws_path)
    eval_data_loader.serialize_examples()

    export_dicts_helper(eval_data_loader.word2index, dicts_dir="corpus", type="vocab",
                        idx2token_path="index2word", token2idx_path="word2index", write_pickle=True)

