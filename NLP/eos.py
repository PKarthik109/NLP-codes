def is_eos(text: str) -> bool:
    text = text.rstrip()          # ignore trailing blanks
    if not text:                  # empty → lots of blank lines → EOS
        return True

    last = text[-1]
    if last in '?!':              # ends with ? or ! → EOS
        return True
    if last != '.':               # ends with anything else → NOT EOS
        return False

    # last char is '.'
    words = text[:-1].split()     # drop the dot, split words
    if not words:                 # lone “.” → treat as EOS
        return True

    last_word = words[-1].lower()
    if last_word in {'etc', 'et al', 'e.g', 'i.e', 'mr', 'mrs', 'dr'}:
        return False              # abbreviation → NOT EOS
    return True                   # ordinary sentence-ending period → EOS


# quick demo
tests = [
    "",                     # blank
    "Hello?",               # ?
    "Hello!",               # !
    "Hello,",               # ,
    "Hello",                # no punctuation
    "Hello.",               # plain period
    "Read books etc.",      # abbreviation
    "See Dr. Smith.",       # another abbreviation
]

for t in tests:
    print(repr(t), "→ EOS" if is_eos(t) else "NOT EOS")