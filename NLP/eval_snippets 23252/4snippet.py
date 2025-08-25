import spacy
Nlp=spacy.load("en_code_web_sm")
Doc=Nlp("cats are running quickly.")
for token in Doc:
         print(token.lemma_)
