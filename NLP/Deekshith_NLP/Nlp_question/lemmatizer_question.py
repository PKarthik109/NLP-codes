#import the lemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()
words=["running","flies","better"]
for word in words :
  print(f"{word}->{lemmatizer.lemmatize(word,pos='v')}")
