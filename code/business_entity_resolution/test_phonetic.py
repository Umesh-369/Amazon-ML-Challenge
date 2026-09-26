import csv, os, sys, time, string, unicodedata, re
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Simplified fast Double Metaphone / Soundex for business names
def fast_soundex(word: str) -> str:
    if not word:
        return ""
    w = word.lower()
    first = w[0]
    table = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    encoded = [first]
    for char in w[1:]:
        code = table.get(char, '')
        if code and code != encoded[-1]:
            encoded.append(code)
    res = "".join(encoded).replace(first, first, 1)
    return (res + "0000")[:4]

print("Testing soundex on Indian variations:")
for p1, p2 in [("aggarwal", "agarwal"), ("chowdhury", "chaudhary"), ("prasad", "prashad"), ("centre", "center")]:
    print(f"  {p1} ({fast_soundex(p1)}) == {p2} ({fast_soundex(p2)}): {fast_soundex(p1) == fast_soundex(p2)}")
