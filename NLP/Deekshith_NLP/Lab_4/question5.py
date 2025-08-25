import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

excel_path = r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx'
df = pd.read_excel(excel_path, sheet_name='Sheet1', engine='openpyxl')
Documents = [ word_tokenize(str(txt)) for txt in df['Report'] ]

stop_words  = set(stopwords.words('english'))
punctuation = set(string.punctuation)

normalized_docs = []
for tokens in Documents:
    clean = [t.lower() for t in tokens
             if t.lower() not in stop_words
                and t not in punctuation]
    normalized_docs.append(clean)

vocab = sorted({tok for doc in normalized_docs for tok in doc})
M = len(vocab)

N = len(normalized_docs)
V2 = np.zeros((N, M), dtype=int)

for i, doc in enumerate(normalized_docs):
    counts = {}
    for tok in doc:
        counts[tok] = counts.get(tok, 0) + 1
    for j, term in enumerate(vocab):
        V2[i, j] = counts.get(term, 0)

cos_sim_matrix = cosine_similarity(V2)

doc_labels = [f"S{i}" for i in range(1, N+1)]
cosine_df = pd.DataFrame(cos_sim_matrix, index=doc_labels, columns=doc_labels)

print("Cosine‐similarity between documents:")
print(cosine_df)