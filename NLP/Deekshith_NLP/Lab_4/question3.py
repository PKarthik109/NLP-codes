import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_excel(r'C:\Users\year3\Desktop\NLP\Lab_4\Data.xlsx', sheet_name='Sheet1', engine='openpyxl')

documents = df['Report'].astype(str).tolist()

Documents = [word_tokenize(doc) for doc in documents]

print(f'Total students tokenized: {len(Documents)}')
for i in range(37):
    print(f'tokens for S{i+1}:',Documents [i][:])
    print('\n')

token_population = list(set([token for doc in Documents for token in doc]))


stop_words = set(stopwords.words('english'))

bag_of_words = [token for token in token_population if token.lower() not in stop_words]

print(f'Length of bag-of-words (after removing stopwords): {len(bag_of_words)}')
print(f'Bag-of-words tokens:\n{bag_of_words}')
