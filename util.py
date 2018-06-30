from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm
from nltk import FreqDist
import re
import numpy as np

tqdm.pandas()


def read_data():
    """
    Read the english and hebrew translations into a dataframe
    :return: Dataframe with the matched sentences
    """
    df = pd.read_csv("data/eng_heb_sentences.csv", encoding="utf8", names=["hebrew_sentences", "english_sentences"],
                     header=0)
    df["english_sentences"] = df["english_sentences"].apply(lambda x: x.lower())
    return df


def decontracted(phrase):
    """
    Change words abbreviations into their actual word meaning
    :param phrase: sentence we want to change
    :return: modified sentence
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_vocab(series, addtional_tokens=[], top=None):
    """
    extract the vocabulary out of an array, allow to add additional tokens to the vocabulary and choose only the top n frequent words.
    :param series: array of sentences
    :param addtional_tokens: additional tokens we want to include in the vocabulary
    :param top: top n frequent words we want to include in the vocabulary
    :return: map from a word to its numeric representation and the opposite map
    """
    rev_vocab = addtional_tokens
    freq_vocab = FreqDist()
    for s in tqdm(series):
        freq_vocab.update(word_tokenize(decontracted(s)))
    print("Original vocab size %s" % len(freq_vocab))
    all_words_sorted = sorted(freq_vocab, key=freq_vocab.get, reverse=True)
    top_words = all_words_sorted[:top]
    rev_vocab += top_words
    vocab = {word: index for index, word in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sentences(sentences, vocab, add_prefix_token=None, add_suffix_token=None, encode=False, reverse=False):
    """
    create a vector out of a sentence
    :param sentences: array of sentences
    :param vocab: the vocabulary that will map the words into numeric representation
    :param add_prefix_token: a token to add as prefix to the sentence vector
    :param add_suffix_token: a token to add as suffix to the sentence vector
    :param encode: if to encode the words into numeric representation
    :param reverse: if to reverse the sentence vector
    :return: array of vectorized sentences
    """
    vectorized_sentences = []
    for s in tqdm(sentences):
        sentence = []
        for word in word_tokenize(decontracted(s)):
            if word in vocab:
                if encode:
                    sentence.append(vocab[word])
                else:
                    sentence.append(word)
            else:
                if encode:
                    sentence.append(vocab["<UNK>"])
                else:
                    sentence.append("<UNK>")
        if add_prefix_token is not None:
            if encode:
                sentence = [vocab[add_prefix_token]] + sentence
            else:
                sentence = [add_prefix_token] + sentence
        if add_suffix_token is not None:
            if encode:
                sentence = sentence + [vocab[add_suffix_token]]
            else:
                sentence = sentence + [add_suffix_token]
        if reverse:
            vectorized_sentences.append(np.array(sentence[::-1]))
        else:
            vectorized_sentences.append(np.array(sentence))
    return np.array(vectorized_sentences)


if __name__ == '__main__':
    pass
    # df = read_data()
    #  # df = clean_english_sentences(df)
    #  eng_vocab, rev_eng_vocab = get_vocab(df["english_sentences"], addtional_tokens=["<UNK>"], top=15000)
    #  heb_vocab, rev_heb_vocab = get_vocab(df["hebrew_sentences"], addtional_tokens=["<UNK>","<START>","<EOS>"], top=30000)
    #  vect_eng_sentences = vectorize_sentences(df["english_sentences"], eng_vocab)
    #  vect_heb_sentences = vectorize_sentences(df["hebrew_sentences"], heb_vocab)
    # pass
