import spacy
nlp=spacy.load("en_core_web_sm")
text="cats are running quickly"
doc=nlp(text)
for token in doc:
    print(token.text,"->",token.lemma_)
