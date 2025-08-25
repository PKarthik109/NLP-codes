import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Load data from Excel file - adjust path if needed
file_path = r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx'  # change this path if file is somewhere else
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Extract the reports as list of documents (37 students)
documents = df['Report'].tolist()

# Question 1: Tokenization of all documents, store as nested list
Documents = [word_tokenize(doc.lower()) for doc in documents]
print(f'Total students tokenized: {len(Documents)}')
for i in range(37):
    print(f'tokens for S{i+1}:',Documents [i][:])
    print('\n')


nltk.download('punkt')

df = pd.read_excel(r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx', sheet_name='Sheet1', engine='openpyxl')

documents = df['Report'].astype(str).tolist()

Documents = [word_tokenize(doc) for doc in documents]

print(f'Total students tokenized: {len(Documents)}')
for i in range(37):
    print(f'tokens for S{i+1}:',Documents [i][:])
    print('\n')


# Question 2: Token population - merge tokens from all documents, distinct tokens list
token_population = list(set([token for doc in Documents for token in doc]))

# Print length of token population
V = len(token_population)
print("Length of token population (V):", V)

# Question 3: Bag-of-Words - remove stopwords and duplicates
stop_words = set(stopwords.words('english'))
bag_of_words = sorted(list(set([token for token in token_population if token.isalpha() and token not in stop_words])))

# Question 4: Document Vectorization
# V1d_w = presence(0/1), V2d_w = count of tokens from bag_of_words

# Create presence and count vectors for each document
V1_vectors = []
V2_vectors = []

for doc_tokens in Documents:
    # create presence vector
    presence_vector = [1 if word in doc_tokens else 0 for word in bag_of_words]
    V1_vectors.append(presence_vector)
    
    # create count vector
    count_vector = [doc_tokens.count(word) for word in bag_of_words]
    V2_vectors.append(count_vector)

# Convert lists to numpy arrays for convenience
V1_vectors = np.array(V1_vectors)
V2_vectors = np.array(V2_vectors)

# Question 5: Calculate cosine similarity between documents using second feature vector (counts)
cosine_sim_matrix = cosine_similarity(V2_vectors)

# Print the cosine similarity matrix shape and a snippet
print("Cosine similarity matrix shape:", cosine_sim_matrix.shape)
print("Cosine similarity matrix snippet (5x5):\n", cosine_sim_matrix[:5,:5])
#QUESTION 4


