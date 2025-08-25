import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paragraph text
paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""

# Preprocessing
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# Tokenization
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Remove stopwords
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=300, min_count=1)

# ✅ Get vocabulary
words = model.wv.index_to_key

# Word Vector Example
vector = model.wv['war']

# Most similar words
similar = model.wv.most_similar('vikram')

print("Vocabulary size:", len(words))
print("Vector for 'war':", vector[:10])  # first 10 dims
print("Most similar to 'vikram':", similar)


# -------------------------
# WORD SIMILARITY
# -------------------------
word1, word2, word3 = "war", "freedom", "india"

vec1 = model.wv[word1]
vec2 = model.wv[word2]
vec3 = model.wv[word3]

sim1 = cosine_similarity([vec1], [vec2])[0][0]
sim2 = cosine_similarity([vec1], [vec3])[0][0]

print("\n--- Word Similarity ---")
print(f"Similarity ({word1}, {word2}): {sim1:.4f}")
print(f"Similarity ({word1}, {word3}): {sim2:.4f}")


# -------------------------
# SENTENCE SIMILARITY (Method 1: Avg Word2Vec)
# -------------------------
def sentence_vector(sentence, model, vector_size=300):
    """Compute avg Word2Vec vector for a sentence"""
    words = [w for w in nltk.word_tokenize(sentence.lower()) if w in model.wv]
    if len(words) == 0:
        return np.zeros(vector_size)
    return np.mean([model.wv[w] for w in words], axis=0)


sent1 = "india got its first vision of freedom"
sent2 = "india must stand up to the world"

vec_s1 = sentence_vector(sent1, model)
vec_s2 = sentence_vector(sent2, model)

sim_sent_avg = cosine_similarity([vec_s1], [vec_s2])[0][0]

print("\n--- Sentence Similarity (Avg Word2Vec) ---")
print(f"Cosine Similarity ({sent1} , {sent2}): {sim_sent_avg:.4f}")


# -------------------------
# SENTENCE SIMILARITY (Method 2: Word Mover’s Distance) optional
# -------------------------
tokens_s1 = [w for w in nltk.word_tokenize(sent1.lower()) if w in model.wv]
tokens_s2 = [w for w in nltk.word_tokenize(sent2.lower()) if w in model.wv]

wmd_distance = model.wv.wmdistance(tokens_s1, tokens_s2)

print("\n--- Sentence Similarity (Word Mover’s Distance) ---")
print(f"WMD Distance ({sent1} , {sent2}): {wmd_distance:.4f}")
