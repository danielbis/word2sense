import os

import time

from tf_records_helper import RecordPrep
from utils import Lang, DataLoader, load_related
from my_cosine import my_cosine, my_mask
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()


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
        self.lstm = None
        if self.use_cuda:
            print("Using CUDA GPU.")
            # stateful=False by default
            self.lstm = tf.keras.layers.CuDNNLSTM(units=self.hidden_size,
                                                  return_sequences=True,
                                                  return_state=True)
        else:
            print("CUDA GPU unavailable, using CPU.")
            self.lstm = tf.keras.layers.LSTM(units=self.hidden_size,
                                             return_sequences=True,
                                             return_state=True)

    def call(self, _input):
        outputs, h_state, c_state = self.lstm(_input)

        return outputs, h_state, c_state


def train(embedding_matrix,
          related_matrix,
          dataset,
          encoder_hidden_size,
          encoder_embedding_size,
          checkpoint_dir,
          _batch_size=16,
          _epochs=10,
          max_seq_len=128,
          normalize_by_related_count=True):
    """

    :param embedding_matrix: tensorflow variable [vocab_size x encoder_embedding_size]
    :param related_matrix: tensorflow variable [on_senses_count x 128]
    :param dataset: tf.data.TFRecordDataset
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
    optimizer = tf.train.AdamOptimizer()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder)

    for epoch in range(0,_epochs):
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
                outputs, hidden, cell = encoder(encoder_input)
                print("outputs shape: ",outputs.shape)
                _outs = tf.transpose(outputs, perm=[1, 0, 2])
                print("_outs shape: ", _outs.shape)

                for i in range(0, encoder_input.shape[1]):
                    print("_outs[i] shape: ", _outs[i].shape)
                    print("related_embs[i] shape: ", related_embs[i].shape)
                    _loss = my_cosine(_outs[i], related_embs[i])
                    _loss = my_mask(related_[i], _loss, batch_size=_batch_size, max_related=128)
                    print("_loss.shape: ", _loss.shape[0])
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
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                epoch_loss))  # could divide epoch loss by  number of batches
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



