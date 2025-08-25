contradiction={
    "ain't":"am not",
    "aren't":"are not",
    "see'ya":"lets meet some othertime",
    "how'r":"how are you?",
    "can'we":"can we meet"}

text="how'r can'we ,i am fine see'ya"
for word in text.split():
    if word.lower() in contradiction:
        text=text.replace(word,contradiction[word.lower()])
        
print(text)