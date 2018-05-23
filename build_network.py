# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:14:41 2018

@author: Lenovo
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
from util import read_data, get_vocab, vectorize_sentences,indent_sentences

df = read_data()
eng_vocab, rev_eng_vocab = get_vocab(df["english_sentences"], addtional_tokens=["<UNK>"], top=15000)
heb_vocab, rev_heb_vocab = get_vocab(df["hebrew_sentences"], addtional_tokens=["<UNK>"], top=30000)
vect_eng_sentences = vectorize_sentences(df["english_sentences"], eng_vocab)
vect_heb_sentences = vectorize_sentences(df["hebrew_sentences"], heb_vocab)
decoder_input_data,decoder_target_data = indent_sentences(vect_heb_sentences)
    
vocab_size_english = len(eng_vocab)
vocab_size_hebrew = len(heb_vocab)
emb_dim = 300

encoder_inputs = Input(shape=(None,))
emb_layer = Embedding(vocab_size_english, emb_dim)(encoder_inputs)
lstm_enc_1 = LSTM(emb_dim, return_state=False)(emb_layer)
lstm_enc_2, state_h, state_c = LSTM(emb_dim, return_state=True)(lstm_enc_1)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
emb_layer2 = Embedding(vocab_size_hebrew, emb_dim)(decoder_inputs)
lstm_dec_1 = LSTM(emb_dim, return_sequences=False)(emb_layer2, initial_state=encoder_states)
lstm_dec_2 = LSTM(emb_dim, return_sequences=True)(lstm_dec_1)

decoder_outputs = Dense(vocab_size_hebrew, activation='softmax')(lstm_dec_2)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([vect_eng_sentences, decoder_input_data],decoder_target_data,
          batch_size=32,
          epochs=10,
          validation_split=0.2)
