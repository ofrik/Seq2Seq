from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from util import read_data, get_vocab, vectorize_sentences
from keras import initializers
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 50  # Latent dimensionality of the encoding space.
num_samples = 2000  # Number of samples to train on.
emb_dim = 50
# Path to the data txt file on disk.
SENTENCE_START_TOKEN = "<START>"
SENTENCE_END_TOKEN = "<EOS>"
OOV_TOKEN = "<UNK>"

df = read_data()[:num_samples]

eng_vocab, rev_eng_vocab = get_vocab(df["english_sentences"], addtional_tokens=[OOV_TOKEN], top=15000)
heb_vocab, rev_heb_vocab = get_vocab(df["hebrew_sentences"],
                                     addtional_tokens=[OOV_TOKEN, SENTENCE_START_TOKEN, SENTENCE_END_TOKEN],
                                     top=15000)

encoder_input_data = np.array(vectorize_sentences(df["english_sentences"], eng_vocab, encode=False))
decoder_input_data = np.array(
    vectorize_sentences(df["hebrew_sentences"], heb_vocab, add_prefix_token=SENTENCE_START_TOKEN,
                        add_suffix_token=SENTENCE_END_TOKEN, encode=False))
# decoder_target_data = np.array([
#     np.concatenate((x[1:], [heb_vocab[SENTENCE_END_TOKEN]]), axis=0) for x in decoder_input_data])
# Vectorize the data.
input_texts = encoder_input_data
target_texts = decoder_input_data

input_characters = rev_eng_vocab
target_characters = rev_heb_vocab
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = eng_vocab
target_token_index = heb_vocab

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# reverse the input
encoder_input_data = np.flip(encoder_input_data, 1)

encoder_inputs = Input(shape=(None,), name="enc_input")
enc_emb = Embedding(num_encoder_tokens, emb_dim)(encoder_inputs)
encoder = LSTM(latent_dim, return_sequences=True, kernel_initializer=initializers.uniform(-0.08, 0.08))
encoder_outputs = encoder(enc_emb)
encoder = LSTM(latent_dim, return_state=True, kernel_initializer=initializers.uniform(-0.08, 0.08))
encoder_outputs, state_h, state_c = encoder(encoder_outputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(num_decoder_tokens, emb_dim)
final_dex = dec_emb(decoder_inputs)
decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, kernel_initializer=initializers.uniform(-0.08, 0.08))
decoder_outputs = decoder_lstm_1(final_dex,
                                 initial_state=encoder_states)
decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True,
                      kernel_initializer=initializers.uniform(-0.08, 0.08))
decoder_outputs, _, _ = decoder_lstm_2(decoder_outputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', kernel_initializer=initializers.uniform(-0.08, 0.08))
decoder_outputs = decoder_dense(decoder_outputs)


def step_decay(epoch):
    initial_lrate = 0.7
    lrate = initial_lrate
    denominator_exp = epoch - 5
    if epoch >= 5:
        lrate = lrate / (2 ^ denominator_exp)
    return lrate


lrate = LearningRateScheduler(step_decay)

callbacks_list = [lrate]

optimizer = SGD(clipnorm=5.0)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_list,
          validation_split=0.2)

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2 = dec_emb(decoder_inputs)
# decoder_outputs2 = decoder_lstm_1(final_dex2, initial_state=decoder_states_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm_2(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

decoder_model.summary()


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index[SENTENCE_START_TOKEN]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = rev_heb_vocab[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == SENTENCE_END_TOKEN or
                    len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(len(input_texts)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)
