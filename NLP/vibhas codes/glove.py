import nltk
import re
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Download GloVe (100d)
# -----------------------
glove_model = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe

# -----------------------
# Paragraph text
# -----------------------
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

# -----------------------
# Preprocessing
# -----------------------
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# Sentence tokenization
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# -----------------------
# Example: Word Vector & Similar Words
# -----------------------
vector = glove_model['war']
print("Vector for 'war':", vector[:10])  # first 10 dimensions

similar = glove_model.most_similar('vikram')
print("\nMost similar to 'vikram':", similar)

print("\nVocabulary size in GloVe:", len(glove_model.key_to_index))

# -----------------------
# Word Similarity
# -----------------------
word1, word2, word3 = "india", "freedom", "vikram"

vec1 = glove_model[word1]
vec2 = glove_model[word2]
vec3 = glove_model[word3]

sim1 = cosine_similarity([vec1], [vec2])[0][0]
sim2 = cosine_similarity([vec1], [vec3])[0][0]

print("\n--- Word Similarity (GloVe) ---")
print(f"Similarity({word1}, {word2}): {sim1:.4f}")
print(f"Similarity({word1}, {word3}): {sim2:.4f}")

# -----------------------
# Sentence Vector Function
# -----------------------
def sentence_vector(sentence, model, dim):
    words = [w for w in nltk.word_tokenize(sentence.lower()) if w in model.key_to_index]
    if len(words) == 0:
        return np.zeros(dim)
    return np.mean([model[w] for w in words], axis=0)

# -----------------------
# Sentence Similarity
# -----------------------
sent1 = "india got its first vision of freedom"
sent2 = "india must stand up to the world"

vec_s1 = sentence_vector(sent1, glove_model, 100)
vec_s2 = sentence_vector(sent2, glove_model, 100)

sim_sent = cosine_similarity([vec_s1], [vec_s2])[0][0]

print("\n--- Sentence Similarity (GloVe) ---")
print(f"Cosine Similarity('{sent1}' , '{sent2}'): {sim_sent:.4f}")
