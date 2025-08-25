import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import re

# Download stopwords if not already done
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------
# Sample text
# -----------------------
text = "Natural language processing enables computers to understand human language."

# -----------------------
# Preprocessing
# -----------------------
text = text.lower()
text = re.sub(r'\d+', '', text)  # remove numbers
text = re.sub(r'\s+', ' ', text)  # remove extra spaces

# Tokenization
tokens = nltk.word_tokenize(text)

# Remove stopwords
tokens = [word for word in tokens if word not in stopwords.words('english')]

print("Tokens:", tokens)

# -----------------------
# Generate n-grams
# -----------------------
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Unigrams
unigrams = generate_ngrams(tokens, 1)
print("\n--- Unigrams ---")
print(unigrams)

# Bigrams
bigrams = generate_ngrams(tokens, 2)
print("\n--- Bigrams ---")
print(bigrams)

# Trigrams
trigrams = generate_ngrams(tokens, 3)
print("\n--- Trigrams ---")
print(trigrams)
