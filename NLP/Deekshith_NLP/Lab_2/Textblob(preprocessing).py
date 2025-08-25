from textblob import TextBlob
text="Barack Obama was born in Hawaii on August 4th,1961.He served as the 44th president of United States."
blob=TextBlob(text)
#Tokenization
tokens=blob.words
#POS tagging
pos_tags=blob.tags
#NER tagging
ner_tags=blob.noun_phrases
print("Tokens :",tokens)
print("POS tags",pos_tags)
print("NER tags",ner_tags)