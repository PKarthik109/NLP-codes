import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

text="i need to write a program in nltk that breaks a corpus(a large collection of txt files) into unigrams and bigrams,i need to write a program in nltk that breaks a corpus"

token=nltk.word_tokenize(text)
unigrams=ngrams(token,3)
print(Counter(unigrams))