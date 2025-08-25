import spacy
nlp = spacy.load("en_core_web_sm")
text="the dgs are barking loudly outside. i am reading a book"
doc=nlp(text)
cleaned_text=[]
for token in doc:
    if not token.is_stop and not token.is_punct:
        lemma = token.lemma_
        cleaned_text.append(lemma.lower())

    cleaned_text=" ".join(cleaned_text)
    print("original text:",text)
    print("preprocessed text",cleaned_text)