
def eos(s):
    s = s.rstrip()
    if not s:
        return True
    if s[-1] in '?!':
        return True
    if s[-1] != '.':
        return False
    # ends with .
    if s.lower().endswith(' etc.'):
        return False
    return True


print(eos("hello,"))