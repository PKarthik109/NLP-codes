import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#intialize Porter Stemmer
Example_text="Hello Mr.Smith ,how are you? How is your pet cats ? How are your children? where are they . are they playing"
stemmer=PorterStemmer()
print(stemmer.stem(Example_text))
#stem each token
print(stemmer.stem("cats"))
print(stemmer.stem("played"))
print(stemmer.stem("playing"))
print(stemmer.stem("welcomes"))
print(stemmer.stem("persual"))
print(stemmer.stem("ideologies"))