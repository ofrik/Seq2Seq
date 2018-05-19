# Seq2Seq
We will implement Seq2Seq algorithm in python like it was implemented in the paper "[Sequence ro Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)"
instead of using english/french dataset we will use english/hebrew one from [tatoeba project](https://tatoeba.org/eng).
We linked the english sentences with the different translations in hebrew into a single csv file that we will use as our dataset.
<!-- Using the tmx files from http://opus.nlpl.eu/ of english/hebrew without OpenSubtitles duplications. -->
<!-- After quick data exploration we could see some of the translations are less than correct (probably data from OpenSubtitles that has alignment issue). -->
<!-- In order to deal with that we used an external translation library TextBlob that has translation api that uses google translate. -->
<!-- Than we manually checked and cleaned the sentences that was not being translated by the 3rd party API. -->
<!-- With the remaining sentences we checked their hebrew translation from the original dataset and the translation of the 3rd party API and checked the cosine similarity. If it was below 0.8 we removed the row. -->


