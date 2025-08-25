from  nltk.stem import PorterStemmer
Stemmer=PorterStemmer()
Words=["running","jumps","easily","fairly"]
for w in Words:
      print( Stemmer.stem(w))
