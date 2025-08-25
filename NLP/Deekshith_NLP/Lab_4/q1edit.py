import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Load Excel
df = pd.read_excel(r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx', sheet_name='Sheet1')

# Extract all reports
documents = df['Report'].tolist()

# Tokenize each document
Documents = [word_tokenize(doc.lower()) for doc in documents]

# Output: Documents (list of lists of tokens)
Documents
