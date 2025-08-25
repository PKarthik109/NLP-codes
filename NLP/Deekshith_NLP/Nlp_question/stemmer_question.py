from nltk.stem import PorterStemmer
Stemmer= PorterStemmer()
Words=["running","jumps","easily","fairly"]
for word in Words:
         print(f"{word}->{Stemmer.stem(word)}")
