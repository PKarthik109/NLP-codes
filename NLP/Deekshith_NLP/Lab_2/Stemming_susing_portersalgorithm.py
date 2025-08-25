import nltk
from nltk.stem import PorterStemmer
#intialize Porter Stemmer
stemmer=PorterStemmer()

#stem each token
print(stemmer.stem("cats"))
print(stemmer.stem("played"))
print(stemmer.stem("playing"))
print(stemmer.stem("welcomes"))
print(stemmer.stem("persual"))
print(stemmer.stem("ideologies"))