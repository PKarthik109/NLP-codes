from textblob import TextBlob

text="Barack Obama was born in Hawai on aug 4 1961 . He served as the 44th President of united states."
blob=TextBlob(text)
tokens=blob.words
pos_tags=blob.tags
ner_tags=blob.noun_phrases
print("tokens:",tokens)
print("pos tags:",pos_tags)
print("ner tags:",ner_tags)