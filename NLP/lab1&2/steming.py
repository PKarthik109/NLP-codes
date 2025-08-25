import nltk
from nltk.stem import PorterStemmer
#initilize porter stemmer
stemmer = PorterStemmer()

print(stemmer.stem("cats"))
print(stemmer.stem("played"))
print(stemmer.stem("playing"))
print(stemmer.stem("welcomes"))
print(stemmer.stem("persual"))
print(stemmer.stem("idealogies"))
print(stemmer.stem("kingdomes"))
print(stemmer.stem("adjustable"))