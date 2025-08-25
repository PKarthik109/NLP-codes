import spacy
#load english tokenizer,tagger,parser,NER, and word vectors
nlp=spacy.load("en_core_web_sm")
text="The dogs are barking loudly outside. I am reading a book"
doc=nlp(text)
cleaned_text=[]
for token in doc:
    #remove stopwords and punctutation
    if not token.is_stop and not token.is_punct:
        #lemmatize each token
        lemma=token.lemma_
        cleaned_text.append(lemma.lower())
cleaned_text=" ".join(cleaned_text)
print("Original text:",text)
print("preprocessed text:",cleaned_text)

