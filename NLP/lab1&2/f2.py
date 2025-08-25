from nltk.tokenize import sent_tokenize,word_tokenize
EXAMPLE_TEXT = "Hello.hi Mr.smith,how are you doing today? The wheater is great, and Python is awesome. The sky is pinkish-Blue. You shouldn't eat cardboard and understand and helloworld heyhibye ksdfojosgdfiohporsdjjwporkigpjp"

print(sent_tokenize(EXAMPLE_TEXT))

print("\n \n")

print(word_tokenize(EXAMPLE_TEXT))