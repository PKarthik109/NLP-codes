#learn unigram aand bigr4am probabbility and print thhe top 5 unigram and bigrams corresponding to 
from collections import Counter
import re

text = "this is a sample text this text is a sample example"

tokens = re.findall(r'\w+', text.lower())

unigram_counts = Counter(tokens)
total_unigrams = sum(unigram_counts.values())

bigrams = list(zip(tokens[:-1], tokens[1:]))
bigram_counts = Counter(bigrams)

unigram_probs = {word: count / total_unigrams for word, count in unigram_counts.items()}

bigram_probs = {bg: count / unigram_counts[bg[0]] for bg, count in bigram_counts.items()}

top_5_unigrams = sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 Unigrams:")
for word, prob in top_5_unigrams:
    print(f"{word}: {prob:.3f}")

print("\nTop 5 Bigrams:")
for (w1, w2), prob in top_5_bigrams:
    print(f"{w1} {w2}: {prob:.3f}") 