import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
text="I need to write a program in NLTK that breaks a corpus (a large collection of txt files) into unigrams and bigrams. I need to write a program in NLTK that breaks a corpus"
token=nltk.word_tokenize(text)
unigrams=ngrams(token,2)
print(Counter(unigrams))
