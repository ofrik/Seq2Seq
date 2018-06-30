
# coding: utf-8

# ### Imports

from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
from util import read_data, get_vocab, vectorize_sentences
import numpy as np
import tensorflow as tf
import os
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
tf.logging.set_verbosity(tf.logging.INFO)


# Starting the tensorflow session
sess = tf.Session()


# Defining the special token that will be used
SENTENCE_START_TOKEN = "<START>"
SENTENCE_END_TOKEN = "<EOS>"
OOV_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"


# Reading the data
df = read_data()


# Spliting the data into train, validation and test
msk = np.random.rand(len(df)) < 0.8
df_train = df[msk]
msk2 = np.random.rand(len(df_train)) < 0.8
df_validate = df_train[~msk2]
df_train = df_train[msk2]
df_test = df[~msk]


print("Train size: %s"%len(df_train))
print("Validation size: %s"%len(df_validate))
print("Test size: %s"%len(df_test))


# Creating the english and hebrew vocabularies
eng_vocab, rev_eng_vocab = get_vocab(df["english_sentences"], addtional_tokens=[PAD_TOKEN,OOV_TOKEN], top=None)
heb_vocab, rev_heb_vocab = get_vocab(df["hebrew_sentences"],
                                         addtional_tokens=[PAD_TOKEN,OOV_TOKEN, SENTENCE_START_TOKEN, SENTENCE_END_TOKEN],
                                         top=None)


def vectorize_dataset(df):
    """
    vectorizing the data into encoder input, decoder input and decoder target.
    """
    vect_eng_sentences = vectorize_sentences(df["english_sentences"], eng_vocab,encode=True,reverse=True)
    decoder_input_data = vectorize_sentences(df["hebrew_sentences"], heb_vocab, add_prefix_token=SENTENCE_START_TOKEN,encode=True)
    decoder_target_data = np.array([np.concatenate((x[1:], [heb_vocab[SENTENCE_END_TOKEN]]), axis=0) for x in decoder_input_data])
    return vect_eng_sentences,decoder_input_data,decoder_target_data


# ### Train data

# Vectorization of the training data
train_vect_eng_sentences,train_decoder_input_data,train_decoder_target_data = vectorize_dataset(df_train)


# ### Validation data

# Vectorization of the validation data
val_vect_eng_sentences,val_decoder_input_data,val_decoder_target_data = vectorize_dataset(df_validate)


# ### Test data

# Vectorization of the test data
test_vect_eng_sentences,test_decoder_input_data,test_decoder_target_data = vectorize_dataset(df_test)


# Getting general data on the data and vocabulary, setting the layers sizes and batch sizes
vocab_size_english = len(eng_vocab)
vocab_size_hebrew = len(heb_vocab)
max_encoder_seq_length = max([len(txt) for txt in np.append(np.append(train_vect_eng_sentences,val_vect_eng_sentences),test_vect_eng_sentences)])
max_decoder_seq_length = max([len(txt) for txt in np.append(np.append(train_decoder_input_data,val_decoder_input_data),test_decoder_input_data)])
emb_dim = 64
lstm_size = 64
batch_size = 128


def pad_sequences(sequence,pad_id,to_length):
    """
    padding the sequences to the max number of words in a vector
    """
    need_to_add = to_length-len(sequence)
    return np.concatenate((sequence,np.array([pad_id]*need_to_add)),axis=0)


# Padding all the train, validation and test vectors we have
train_vect_eng_sentences = np.array([pad_sequences(sentence,eng_vocab[PAD_TOKEN],max_encoder_seq_length) for sentence in train_vect_eng_sentences])
train_decoder_input_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in train_decoder_input_data])
train_decoder_target_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in train_decoder_target_data])
val_vect_eng_sentences = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_encoder_seq_length) for sentence in val_vect_eng_sentences])
val_decoder_input_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in val_decoder_input_data])
val_decoder_target_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in val_decoder_target_data])
test_vect_eng_sentences = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_encoder_seq_length) for sentence in test_vect_eng_sentences])
test_decoder_target_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in test_decoder_target_data])
test_decoder_input_data = np.array([pad_sequences(sentence,heb_vocab[PAD_TOKEN],max_decoder_seq_length) for sentence in test_decoder_input_data])


# ### inputs and outputs

# Creating the placeholders, the inputs for the model, both encoder, decoder and output
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_inputs_length')


# ### embeddings

# Creating the embeddings layers for both encoder and decoder
embedding_encoder = tf.get_variable("embedding_encoder", [vocab_size_english, emb_dim])

encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

embedding_decoder = tf.get_variable("embedding_decoder", [vocab_size_hebrew, emb_dim])

decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

lstm_initializer = tf.random_uniform_initializer(-0.08,0.08)


# ### encoder

# Creating the two layers of LSTM encoder
encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(lstm_size,initializer=lstm_initializer) for _ in range(2)])

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,sequence_length=encoder_inputs_length,dtype=tf.float32, time_major=True)


# ### decoder

# Creating the two layers of LSTM decoder
decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(lstm_size,initializer=lstm_initializer) for _ in range(2)])

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_emb_inp,initial_state=encoder_state,sequence_length=decoder_inputs_length,
    dtype=tf.float32, time_major=True, scope="plain_decoder",
)


# Transfer the logits into an actual word prediction
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size_hebrew)

decoder_prediction = tf.argmax(decoder_logits, 2)


# ### loss and train

# Calculate the cross entropy according to the logits and labels and the loss from that. Also initializing the optimizer and cliping the gradients like in the paper
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size_hebrew, dtype=tf.float32),
                                                                 logits=decoder_logits)
loss = tf.reduce_mean(stepwise_cross_entropy)
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer()
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)


# merge all the scalar of the summary and creating a tensorboard writter
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('tensorboard/train',sess.graph)


# ### train

# initialize the model variables
sess.run(tf.global_variables_initializer())


# setup a model saver and restore the model if one already exists
saver = tf.train.Saver()

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
latest_model = tf.train.latest_checkpoint("checkpoints")

if latest_model is not None:
    saver.restore(sess, "model.ckpt")


def get_batch(enc_input,dec_input,dec_target,batch_size):
    """
    creating a feed mini batch from the data
    """
    number_of_batches = int(len(enc_input)/batch_size)
    counter = 0
    while True:
        from_index = (counter%number_of_batches)*batch_size
        to_index = ((counter%number_of_batches)+1)*(batch_size)
        feed = {}
        if enc_input is not None:
            feed[encoder_inputs]=enc_input[from_index:to_index].T
            feed[encoder_inputs_length] = np.array([len([word for word in sequence if word!=0]) for sequence in enc_input[from_index:to_index]],dtype=np.int32)
        if dec_input is not None:
            feed[decoder_inputs]=dec_input[from_index:to_index].T
            feed[decoder_inputs_length] = np.array([len([word for word in sequence if word!=0]) for sequence in dec_input[from_index:to_index]],dtype=np.int32)
        if dec_target is not None:
            feed[decoder_targets] = dec_target[from_index:to_index].T
        yield feed
        counter += 1

def decode_sequence(sequence,rev_vocab,to_join=True):
    """
    decode a vector into a sentence
    """
    arr = [rev_vocab[int(index)] for index in sequence if rev_vocab[int(index)]!="<PAD>" and rev_vocab[int(index)]!="<EOS>"]
    if to_join: 
        return " ".join(arr)
    else:
        return arr

def get_bleu_summary(value,iteration):
    """
    create bleu summary value
    """
    bleu_summary = tf.Summary(value=[tf.Summary.Value(tag="validation_bleu", simple_value=value), ])
    return bleu_summary

chencherry = SmoothingFunction()
def get_validation_bleu_scores(sess,generator,iterations):
    """
    calculate the mean bleu score for specific number of minibatch iterations of a generator
    """
    scores = []
    for i in range(iterations):
        feed = next(generator)
        validation_predict_ = sess.run(decoder_prediction, feed)
        scores += [sentence_bleu([decode_sequence(pred,rev_heb_vocab,False)],decode_sequence(exp,rev_heb_vocab,False),smoothing_function=chencherry.method1) for pred,exp in zip(validation_predict_.T,feed[decoder_targets].T)]
    return np.mean(scores)

loss_track = []


# Train the model while output the loss every 50 minibatches, calculate the validation bleu, save and print examples of translations every 250 minibatches
max_batches = 20500
batches_per_epoch = int(len(train_vect_eng_sentences)/batch_size)
batches_per_epoch_validation = int(len(val_vect_eng_sentences)/batch_size)
log_every_iterations = 50
print_samples_every_iterations = 250
train_feed_generator = get_batch(train_vect_eng_sentences,train_decoder_input_data,train_decoder_target_data,batch_size)
validation_feed_generator = get_batch(val_vect_eng_sentences,val_decoder_input_data,val_decoder_target_data,batch_size)
try:
    for batch in range(max_batches):
        if batch%batches_per_epoch==0:
            print("Epoch %s"%(int(batch/batches_per_epoch)))
        fd = next(train_feed_generator)
        _, l,summary = sess.run([train_op, loss,merged], fd)
        train_writer.add_summary(summary, batch)
        loss_track.append(l)
        if batch == 0 or batch % log_every_iterations == 0:
            print('  minibatch loss: {}'.format(l))
            mean_validation_bleu = get_validation_bleu_scores(sess,validation_feed_generator,batches_per_epoch_validation)
            train_writer.add_summary(get_bleu_summary(mean_validation_bleu,batch), batch)
            saver.save(sess,"checkpoints/model",global_step=batch)
            if batch % print_samples_every_iterations == 0:
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred,exp) in enumerate(zip(fd[encoder_inputs].T, predict_.T,fd[decoder_targets].T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(decode_sequence(inp[::-1],rev_eng_vocab)))
                    print('    predicted_raw > {}'.format(pred))
                    print('    expected_raw > {}'.format(exp))
                    print('    predicted > {}'.format(decode_sequence(pred,rev_heb_vocab)))
                    print('    expected > {}'.format(decode_sequence(exp,rev_heb_vocab)))
                    if i >= 2:
                        break
                print()
except KeyboardInterrupt:
    print('training interrupted')

test_feed_generator = get_batch(test_vect_eng_sentences,test_decoder_input_data,test_decoder_target_data,batch_size)

number_of_batches_in_test = int(len(test_vect_eng_sentences)/batch_size)


# Calcualte the bleu of the translations of the test data
test_sentences = []
for i in range(number_of_batches_in_test):
    fd = next(test_feed_generator)
    predict_ = sess.run(decoder_prediction, fd)
    for i, (inp, pred,exp) in enumerate(zip(fd[encoder_inputs].T, predict_.T,fd[decoder_targets].T)):
        input_sentence = decode_sequence(inp[::-1],rev_eng_vocab)
        output_sentence = decode_sequence(pred,rev_heb_vocab)
        expected_sentence = decode_sequence(exp,rev_heb_vocab)
        score = sentence_bleu([decode_sequence(pred,rev_heb_vocab,False)],decode_sequence(exp,rev_heb_vocab,False),smoothing_function=chencherry.method1)
        test_sentences.append([input_sentence,output_sentence,expected_sentence,score])


# Save the test results to a CSV
import pandas as pd
test_df = pd.DataFrame(test_sentences,columns=["english_sentence","predicted_translation","expected_translation","bleu_score"])

test_df.to_csv("test_results.csv",encoding="utf-8",index=False)

