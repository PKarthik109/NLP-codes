import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

excel_path = r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx'
df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl')

Documents = [ word_tokenize(str(txt)) for txt in df['Report'] ]

stop_words  = set(stopwords.words('english'))
punctuation = set(string.punctuation)

normalized_docs = []
for tokens in Documents:
    clean = [
        t.lower()
        for t in tokens
        if t.lower() not in stop_words
           and t not in punctuation
    ]
    normalized_docs.append(clean)

vocab_set  = {token for doc in normalized_docs for token in doc}
vocab_list = sorted(vocab_set)      

N = len(normalized_docs)        
M = len(vocab_list)

V1 = np.zeros((N, M), dtype=int)    
V2 = np.zeros((N, M), dtype=int)    

for d_idx, doc in enumerate(normalized_docs):
    token_counts = {}
    for token in doc:
        token_counts[token] = token_counts.get(token, 0) + 1

    for w_idx, word in enumerate(vocab_list):
        count = token_counts.get(word, 0)
        V2[d_idx, w_idx] = count
        V1[d_idx, w_idx] = 1 if count > 0 else 0

print("Number of documents (N):", N)
print("Vocabulary size (M):", M)
print("V1 shape:", V1.shape)
print("V2 shape:", V2.shape)

print("\nFirst 10 entries of V1[0]:", V1[0, :10])
print("First 10 entries of V2[0]:", V2[0, :10])

np.save('V1_binary.npy', V1)
np.save('V2_counts.npy', V2)