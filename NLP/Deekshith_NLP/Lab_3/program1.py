import nltk 
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import WordNetLemmatizer 
import string 
with open(r'C:\Users\year3\Desktop\NLP\Lab_3\data1.txt', "r") as file: 
    text = file.read() 
tokens = word_tokenize(text) 
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 
tokens = [word for word in tokens if word not in string.punctuation] 
print("stemmer output:") 

for token in tokens: 
   print(f"{token} -> {stemmer.stem(token)}") 
print("\nlemmatizer output:") 

for token in tokens: 
    print(f"{token} -> {lemmatizer.lemmatize(token)}") 