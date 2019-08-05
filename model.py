import os

import time

from tf_records_helper import RecordPrep
from utils import Lang, DataLoader, load_related
from my_cosine import my_cosine, my_mask
import tensorflow as tf
tf.enable_eager_execution()
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

def get_checkpoint(encoder, optimizer):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder)
    return checkpoint, checkpoint_prefix

def train(embedding_matrix,
          related_matrix,
          encoder_hidden_size,
          encoder_embedding_size,
          checkpoint,
          checkpoint_prefix,
          _batch_size=16,
          _epochs=10,
          max_seq_len=128,
          normalize_by_related_count=True):

    loader = DataLoader()
    dataset = loader.prepare_dataset_iterators(test_data)
    encoder = Encoder(encoder_hidden_size, encoder_embedding_size, _batch_size)
    optimizer = tf.train.AdamOptimizer()

    for epoch in _epochs:
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
                print(outputs.shape)
                _outs = tf.transpose(outputs, perm=[1, 0, 2])
                print(_outs.shape)

                for i in range(0, max_seq_len):
                    _loss = my_cosine(_outs[i], related_embs[i])

                    _loss = my_mask(related_[i], _loss, batch_size=4, max_related=2)
                    _loss = tf.reduce_sum(_loss)
                    if normalize_by_related_count:
                        _loss = tf.math.divide(_loss, _loss.shape[0])
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

if __name__ == "__main__":

    files = os.listdir("/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/tf_records_corpus")
    test_data = []
    for i,f in enumerate(files):
        test_data.append("/Users/daniel/Desktop/Research/WSD_Data/ontonotes-release-5.0/api/tf_records_corpus/" + f)
        if i > 20:
            break


    
    loader = DataLoader()
    train_ds = loader.prepare_dataset_iterators(test_data)
    #print(train_ds)

    for (batch, example) in enumerate(train_ds):

        if batch > 4:
            break
        else:
            print(example['vocab_ids'])

