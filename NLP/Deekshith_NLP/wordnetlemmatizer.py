from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
print("rocks:",lemmatizer.lemmatize("rocks"))
print("corpora:",lemmatizer.lemmatize("corpora"))
print("better",lemmatizer.lemmatize("better"))
print("better",lemmatizer.lemmatize("better",pos='a'))
print("are",lemmatizer.lemmatize("are",pos='v'))
print("is",lemmatizer.lemmatize("is",pos='v'))
print("am",lemmatizer.lemmatize("am",pos='v'))