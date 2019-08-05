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


"""
See also: tf.ragged.boolean_mask, which can be applied to both dense
and ragged tensors, and can be used if you need to preserve the masked 
dimensions of tensor (rather than flattening them, as tf.boolean_mask does).
"""


def my_mask(ids, loss, batch_size, max_related):
    """
      :param ids sense ids list 0 == PAD
      :param loss batch_size x max_related tensor of cosine distances
      :param batch_size
      :param max_related
      :return: masked losses, removed distances to PAD
    """
    _zeros = tf.zeros([batch_size, max_related], tf.int32)
    b = tf.not_equal(ids, _zeros)
    masked = tf.boolean_mask(loss, b)

    return masked