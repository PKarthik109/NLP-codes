import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import string

nltk.download('punkt')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Read input file
with open(r"C:\Users\pkart\OneDrive\Desktop\NLP\LAB 3\data1.txt", "r") as file:
    text = file.read()

# Tokenize
tokens = word_tokenize(text)
k=sent_tokenize(text)

# Remove punctuation tokens
tokens = [word for word in tokens if word not in string.punctuation]


print("Token:", tokens)
print("Stemmed:", " ".join(stemmer.stem(w) for w in tokens))
print("Lemmatized:"," ".join(lemmatizer.lemmatize(w) for w in tokens))
print(k)
