import nltk 
from nltk.tokenize import word_tokenize 
from nltk.util import bigrams 
from collections import Counter 
import string 

 

with open(r'C:\Users\year3\Desktop\NLP\Lab_3\data2.txt', 'r') as file: 
    text = file.read() .lower()

tokens = word_tokenize(text) 
tokens = [word for word in tokens if word not in string.punctuation] 

unigram_counts = Counter(tokens) 
bigram_counts = Counter(bigrams(tokens)) 

total_unigrams = sum(unigram_counts.values()) 
total_bigrams = sum(bigram_counts.values())  

unigram_probs = {word: count / total_unigrams for word, count in unigram_counts.items()} 
bigram_probs = {word: count / unigram_counts[word[0]] for word, count in bigram_counts.items()} 

sorted_unigrams = sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True) [:5]
sorted_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True) [:5]
print("Top 5 Unigrams by Probability:") 

for word, prob in sorted_unigrams[:5]: 
    print(f"{word}: {prob:.4f}") 

print("\nTop 5 Bigrams by Probability:") 

for (w1, w2), prob in sorted_bigrams[:5]: 
    print(f"{w1} {w2}: {prob:.4f}") 

 