import re
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Paragraph text
# -------------------------
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

# -------------------------
# Preprocessing
# -------------------------
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)
text = text.lower()
text = re.sub(r'\d', ' ', text)
text = re.sub(r'\s+', ' ', text)

# -------------------------
# Load Sentence Transformer
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Encode entire paragraph
# -------------------------
paragraph_embedding = model.encode(text)
print("\nParagraph embedding shape:", paragraph_embedding.shape)

# -------------------------
# Word Embeddings & Similarity
# -------------------------
word1, word2 = "vikram", "india"
word_embeddings = {w: model.encode(w) for w in [word1, word2]}

print("\n--- Word Embeddings ---")
print(f"Embedding for '{word1}' (first 10 dims): {word_embeddings[word1][:10]}")
print(f"Embedding for '{word2}' (first 10 dims): {word_embeddings[word2][:10]}")

sim_words = util.cos_sim(word_embeddings[word1], word_embeddings[word2])
print(f"\nSimilarity({word1}, {word2}): {sim_words.item():.4f}")

# -------------------------
# Sentence Embeddings & Similarity
# -------------------------
sent1 = "india got its first vision of freedom"
sent2 = "india must stand up to the world"

sent_embeddings = {s: model.encode(s) for s in [sent1, sent2]}

print("\n--- Sentence Embeddings ---")
print(f"Embedding for sentence 1 (first 10 dims): {sent_embeddings[sent1][:10]}")
print(f"Embedding for sentence 2 (first 10 dims): {sent_embeddings[sent2][:10]}")

sim_sent = util.cos_sim(sent_embeddings[sent1], sent_embeddings[sent2])
print(f"\nCosine Similarity({sent1} , {sent2}): {sim_sent.item():.4f}")

# -------------------------
# Paragraph vs. Sentence Similarity
# -------------------------
sim_para_sent1 = util.cos_sim(paragraph_embedding, sent_embeddings[sent1])
sim_para_sent2 = util.cos_sim(paragraph_embedding, sent_embeddings[sent2])

print(f"\nSimilarity(Paragraph , '{sent1}'): {sim_para_sent1.item():.4f}")
print(f"Similarity(Paragraph , '{sent2}'): {sim_para_sent2.item():.4f}")
