import os
from utils import Lang, DataLoader, EvalDataLoader, load_related
from model import train, validation, Encoder
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

# HYPERPARAMETERS
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.01
GRADIENT_CLIPPING = None
ENCODER_HIDDEN_SIZE = 150
EMBEDDING_SIZE = 300
MAX_SEQ_LEN = 128
CHECKPOINT_DIR = "./training_checkpoints"


# Paths to data
vocab_path = "./corpus/vocab/pickles/index2word.pickle"
sense_path = "./corpus/sense_vocab/pickles/index2sense.pickle"
related_path = "./corpus/related/relations.pickle"
antonyms_path = "./corpus/related/relations_antonyms.pickle"
embeddings_path = "./corpus/embeddings/GoogleNews-vectors-negative300.bin.gz"
path_to_tf_records = "./test_tf"
path_to_valid_records = "./scws_records/valid/scws_valid.tfrecord"
path_to_test_records = "./scws_records/test/scws_test.tfrecord"


# Helper class
lang = Lang(vocab_path, sense_path, related_path, antonyms_path, _embedding_size=300)
lang.load_gensim_word2vec(embeddings_path)  # load pre-trained embeddings
# this creates TF matrix of embeddings
embeddings_matrix = lang.create_embeddings()
# this loads a matrix of on_sense -> related words mappings
related = load_related(related_path)

# load tf_records
tf_records_files = [f for f in os.listdir(path_to_tf_records) if f[0] != "."]
#data_loader = DataLoader()
#dataset = data_loader.make_dataset(tf_records_files, batch_size= BATCH_SIZE)
eval_data_loader = EvalDataLoader()
test_dataset = eval_data_loader.make_dataset([path_to_test_records], batch_size=1)

"""
train(embedding_matrix=embeddings_matrix,
      related_matrix=related,
      dataset=dataset,
      encoder_hidden_size=ENCODER_HIDDEN_SIZE,
      encoder_embedding_size=EMBEDDING_SIZE,
      checkpoint_dir=CHECKPOINT_DIR,
      _batch_size=BATCH_SIZE,
      _epochs=EPOCHS,
      max_seq_len=MAX_SEQ_LEN,
      normalize_by_related_count=True)
"""
encoder = Encoder(ENCODER_HIDDEN_SIZE, EMBEDDING_SIZE, 4)
rho, pvalue = validation(encoder=encoder, embedding_matrix=embeddings_matrix, dataset=test_dataset)
print("RHO: ", rho)
print("PVALUE: ", pvalue)