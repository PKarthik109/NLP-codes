from nltk.stem import WordNetLemmatizer
Lemmatizer=WordNetLemmatizer()
Words=["running","flies","better"]
for w in Words:
      print(Lemmatizer.lemmatize(w))
