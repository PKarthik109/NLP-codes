import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re


# Step 1: Load dataset
df = pd.read_csv("ecommerceDataset.csv")

# Step 2: Rename the long column name to "description"
df = df.rename(columns={df.columns[1]: "description"})

# Step 3: Extract text column
text_data = df['description'].dropna().astype(str).tolist()

# Step 4: Combine into one big paragraph
paragraph = " ".join(text_data)

# Step 5: Preprocessing
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)   # remove references like [1], [23]
text = re.sub(r'\s+', ' ', text)               # remove extra spaces
text = text.lower()                            # lowercase
text = re.sub(r'\d', ' ', text)                # remove digits
text = re.sub(r'\s+', ' ', text)               # remove extra spaces again

# Step 6: Sentence & word tokenization
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Step 7: Remove stopwords
stop_words = set(stopwords.words('english'))
sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]

# Step 8: Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Step 9: Vocabulary words
words = model.wv.index_to_key

# Step 10: Example word vector
if 'product' in words:
    vector = model.wv['product']
    print("Vector for 'product':", vector[:10])

# Step 11: Example: most similar words
if 'customer' in words:
    similar = model.wv.most_similar('customer')
    print("Most similar to 'customer':", similar)

print("Vocabulary size:", len(words))
