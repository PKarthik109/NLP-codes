contractions={
    "ain't":"am not",
    "aren't":"are not ",
    "can't":"cannot",
    "can't've":"cannot have",
    "cause":"because",
    "could've":"could have",
    "couldn't":"could not ",
    "couldn't've":"couldnot have",
    "didn't":"did not",
    "doedn't": "does not",
    "don't":"do not"}
text ="i don't  agree with this"
text1="hello,I couldn't attend the metting cause, I was busy that time ,sorry about that"
text2="hi,I ain't well that time when you have called ,so I couldn't attend the call"
for word in text.split():
    if word.lower() in contractions:
        text=text.replace(word,contractions[word.lower()])
print(text)
for word1 in text1.split():
    if word1.lower() in contractions:
        text1=text1.replace(word1,contractions[word1.lower()])
print(text1)
for word in text2.split():
    if word.lower() in contractions:
        text2=text2.replace(word,contractions[word.lower()])
print(text2)