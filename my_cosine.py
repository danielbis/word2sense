import tensorflow as tf


def my_cosine(_input, _related):
    """
      _input [batch_size x emb_size]
      _related [batch_size x num_related x emb_size]
      return: cosine distance [batch_size x num_related]
    """
    # tf.expand_dims(
    _input = tf.math.l2_normalize(_input, axis=1)
    _related = tf.transpose(tf.math.l2_normalize(_related, axis=2), perm=[0, 2, 1])

    product = tf.einsum('be,ber->br', _input, _related)
    return 1 - product
