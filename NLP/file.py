# import nltk
# from nltk import sent_tokenize,word_tokenize
# from nltk.stem import PorterStemmer

# stemmer = PorterStemmer()

# with open('data.txt', 'r') as file:
#     content = file.read()
#     print(content)

# print("\nAfter stemming: \n")
# print(stemmer.stem(content))

#first tried logic only decapitalises every word(forgot tokenization)

#  below modifed code:

# import nltk
# from nltk import word_tokenize
# from nltk.stem import PorterStemmer

# stemmer = PorterStemmer()

# with open('data.txt', 'r') as file:
#     content = file.read()
#     print(content)


# k = word_tokenize(content)

# print("\nAfter stemming: \n")
# for word in k:
#     print(stemmer.stem(word), end=' ')


import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import string

stemmer = PorterStemmer()

with open('data.txt', 'r') as file:
    content = file.read()
    print("Original Content:\n", content)


tokens = word_tokenize(content)
cleaned_tokens = [word for word in tokens if word not in string.punctuation]

print("\nAfter Stemming:\n")
for word in cleaned_tokens:
    print(stemmer.stem(word), end=' ')