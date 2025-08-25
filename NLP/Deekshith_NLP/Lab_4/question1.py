import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

df = pd.read_excel(r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx', sheet_name='Sheet1', engine='openpyxl')

documents = df['Report'].astype(str).tolist()

Documents = [word_tokenize(doc) for doc in documents]

print(f'Total students tokenized: {len(Documents)}')
for i in range(37):
    print(f'tokens for S{i+1}:',Documents [i][:])
    print('\n')

