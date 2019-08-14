import os

import time

from scipy.stats import spearmanr
import numpy as np
from tf_records_helper import RecordPrep
from utils import Lang, DataLoader, load_related
from my_cosine import my_cosine, my_mask
import tensorflow as tf
import logging
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

#logging.basicConfig(filename='./training.log', level=logging.INFO)
logger = logging.getLogger('train_log')
hdlr = logging.FileHandler('./training.log')
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


class Encoder(tf.keras.Model):
    """
      Encodes using BiLSTM, consider using Transformer or XLNet
    """

    def __init__(self, _hidden_size, _embedding_size, _batch_size=16):
        super(Encoder, self).__init__()
        self.hidden_size = _hidden_size
        self.embedding_size = _embedding_size
        self.batch_size = _batch_size
        self.use_cuda = tf.test.is_gpu_available(cuda_only=True,
                                                 min_cuda_compute_capability=None)
        """
        The layer returns the hidden state for each input time step, 
        then separately, the hidden state output for the last time step 
        nd the cell state for the last input time step.
        """
        self.lstm_layer1 = None
        self.lstm_layer2 = None

        self.initial_state_layer_1 = [tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32)]

        self.initial_state_layer_2 = [tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32),
                                      tf.zeros((self.batch_size, self.hidden_size), dtype=tf.float32)]

        if self.use_cuda:
            print("Using CUDA GPU.")
            logging.info("Using CUDA GPU.")
            # stateful=False by default
            self.lstm_layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=self.hidden_size,
                                                                                       return_sequences=True,
                                                                                       return_state=True),
                                                                                       merge_mode="concat")
            self.lstm_layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(units=self.hidden_size,
                                                                                       return_sequences=True,
                                                                                       return_state=True),
                                                                                       merge_mode="concat")
        else:
            print("CUDA GPU unavailable, using CPU.")
            logging.info("CUDA GPU unavailable, using CPU.")

            self.lstm_layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.hidden_size,
                                                                                  return_sequences=True,
                                                                                  return_state=True),
                                                                                  merge_mode="concat")
            self.lstm_layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.hidden_size,
                                                                                  return_sequences=True,
                                                                                  return_state=True),
                                                                                  merge_mode="concat")

    def call(self, _input, _initial_states):
        """
        Call a two-layer BiLSTM
        :param _input:
        :param _initial_state:
        :return: output of layer two, layer1 cell state list, layer2 cell state list
        """

        if _initial_states is None:
            _initial_states = [self.initial_state_layer_1, self.initial_state_layer_2]

        outputs_1, h_state_f_1, c_state_f_1, h_state_b_1, c_state_b_1 = self.lstm_layer1(_input,
                                                                                    initial_state=_initial_states[0])

        outputs_2, h_state_f_2, c_state_f_2, h_state_b_2, c_state_b_2 = self.lstm_layer2(outputs_1,
                                                                                    initial_state=_initial_states[1])

        return outputs_2, [h_state_f_1, c_state_f_1, h_state_b_1, c_state_b_1],\
            [h_state_f_2, c_state_f_2, h_state_b_2, c_state_b_2]


def train(embedding_matrix,
          related_matrix,
          dataset,
          validation_dataset,
          test_dataset,
          encoder_hidden_size,
          encoder_embedding_size,
          _learning_rate,
          checkpoint_dir,
          _batch_size=16,
          _epochs=10,
          max_seq_len=128,
          normalize_by_related_count=True):
    """

    :param embedding_matrix: tensorflow variable [vocab_size x encoder_embedding_size]
    :param related_matrix: tensorflow variable [on_senses_count x 128]
    :param dataset: tf.data.TFRecordDataset
    :param validation_dataset: tf.data.TFRecordDataset
    :param encoder_hidden_size: int
    :param encoder_embedding_size: int
    :param checkpoint_dir: str, path to checkpoints
    :param _batch_size: int
    :param _epochs: int
    :param max_seq_len: int
    :param normalize_by_related_count: Boolean, if True batch loss is divided by the total number of related words used
        in that batch
    :return:
    """

    encoder = Encoder(encoder_hidden_size, encoder_embedding_size, _batch_size)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=_learning_rate)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder)

    rho_test, pvalue_test, avg_loss = validation(encoder, embedding_matrix=embedding_matrix, dataset=test_dataset)

    print("Initial Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(rho_test, avg_loss))
    logger.info("Initial Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(rho_test, avg_loss))
    # initial_state for Bidirectional wrapper may cause issues in TF version 1.14
    # update to tf nightly to fix it or use TF 2.0
    hidden_cell_zero = [tf.zeros((_batch_size, encoder_hidden_size), dtype=tf.float32),
                        tf.zeros((_batch_size, encoder_hidden_size), dtype=tf.float32)]
    # list of 2 x 4 x tf.zeros((6, 150), dtype=tf.float32)
    lstm_state = [hidden_cell_zero + hidden_cell_zero, hidden_cell_zero + hidden_cell_zero]
    for epoch in range(0, _epochs):
        start = time.time()  # get start of the epoch
        epoch_loss = tf.contrib.eager.Variable(0, dtype=tf.float32)  # get epoch loss

        for (batch, example) in enumerate(dataset):
            batch_loss = 0.0

            # get word embeddings
            encoder_input = tf.nn.embedding_lookup(embedding_matrix, example['vocab_ids'])
            # get ids of related words
            related_ = tf.nn.embedding_lookup(related_matrix, example['sense_ids'])
            # get embeddings of related words and transpose
            related_embs = tf.transpose(
                tf.nn.embedding_lookup(embedding_matrix, related_), perm=[1, 0, 2, 3])
            related_ = tf.transpose(related_, perm=[1, 0, 2])

            with tf.GradientTape() as tape:
                # f = forward, b = backward
                outputs, layer1_state, layer2_state = encoder(encoder_input, lstm_state)
                lstm_state = [layer1_state, layer2_state]
                _outs = tf.transpose(outputs, perm=[1, 0, 2])

                for i in range(0, encoder_input.shape[1]):
                    _loss = my_cosine(_outs[i], related_embs[i])
                    _loss = my_mask(related_[i], _loss, batch_size=_batch_size, max_related=128)
                    # mask function flattens the output, so shape[0] is the total number of related words
                    # used for cosine loss calculations in that step
                    denum = _loss.shape[0]
                    if denum == 0:
                        denum += 1
                    _loss = tf.reduce_sum(_loss)
                    if normalize_by_related_count:
                        _loss = tf.math.divide(_loss, tf.cast(denum, tf.float32))
                    batch_loss = tf.add(batch_loss, _loss)

            epoch_loss = tf.add(epoch_loss, batch_loss)
            variables = encoder.variables

            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
                logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            epoch_loss))  # could divide epoch loss by  number of batches
        logger.info('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            epoch_loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        if epoch % 2:
            rho, pvalue, avg_loss= validation(encoder, embedding_matrix=embedding_matrix, dataset=validation_dataset)
            print("Spearman's rank correlation after Epoch {} is {:.4f}, average validation loss is {:.4f}".format(
                epoch + 1, rho, avg_loss))
            logger.info("Spearman's rank correlation after Epoch {} is {:.4f}, average validation loss is {:.4f}".format(
                epoch + 1, rho, avg_loss))
        if (epoch + 1) % 5 == 0:
            rho_test, pvalue_test, avg_loss = validation(encoder, embedding_matrix=embedding_matrix,
                                                         dataset=test_dataset)

            print("Epoch {} Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(epoch, rho_test,
                                                                                                       avg_loss))
            logger.info("Epoch {} Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(
                epoch, rho_test, avg_loss))

    rho_test, pvalue_test, avg_loss = validation(encoder, embedding_matrix=embedding_matrix, dataset=test_dataset)
    print("Final Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(rho_test, avg_loss))
    logger.info("Final Spearman's rank correlation is {:.4f}, average test loss is {:.4f}".format(rho_test, avg_loss))

def validation(encoder, embedding_matrix, dataset):
    """
    :param encoder: tf.keras.Model, callable
    :param embedding_matrix: matrix storing word embeddings
    :param dataset: tf.data.TFRecordDataset with eval or test data
    :return: rho: Spearman's rank correlation coefficient between cosine distance and human similarity ratings from scws,
             pvalue: the two-sided p-value for a hypothesis test whose null hypothesis
                is that two sets of data are uncorrelated, has same dimension as rho.
    """
    eval_cosines = []
    eval_ratings = []
    num_batches = 0
    for (batch, example) in enumerate(dataset):
        batch_cosines = []
        batch_ratings = []
        # get word embeddings
        encoder_input_1 = tf.nn.embedding_lookup(embedding_matrix, example['sentence1'])
        encoder_input_2 = tf.nn.embedding_lookup(embedding_matrix, example['sentence2'])

        sentence1_out, s_1_layer1_state, s_1_layer2_state = encoder(encoder_input_1, None)  # None for initial state
        sentence2_out, s_2_layer1_state, s_2_layer2_state = encoder(encoder_input_2, None)

        if sentence1_out.shape[0] != sentence2_out.shape[0]:
            raise ValueError(
                'batch_size of sentence1 and batch_size of sentence2 are not equal : '
                'batch_size_1={}, batch_size_2={}'.format(sentence1_out.shape[0], sentence2_out.shape[0]))

        for i in range(0, sentence1_out.shape[0]):
            """
            my_cosine
            _input [batch_size x emb_size]
            _related [batch_size x num_related x emb_size]
            return: cosine distance [batch_size x num_related]
            """
            word1 = sentence1_out[0][example["word1idx"][i]]
            word2 = sentence2_out[0][example["word2idx"][i]]

            loss = my_cosine(tf.expand_dims(word1, 0), tf.expand_dims(tf.expand_dims(word2, 0), 0))
            batch_cosines.append(tf.squeeze(loss))
            batch_ratings.append(example["avg_rating"][i])

        eval_cosines += batch_cosines
        eval_ratings += batch_ratings
        num_batches = batch


    rho, pvalue = spearmanr(eval_cosines, eval_ratings)

    return rho, pvalue, np.sum(eval_cosines) / num_batches
