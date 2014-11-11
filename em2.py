import os
import re
import sys
import math
import numpy
import scipy

REGEX = r'[a-zA-Z]+'
COMPILED_REGEX = re.compile(REGEX)

def get_content(text):
    t = text.split('\n\n', 1)
    return t[1]

def select_features(text):
    features = []
    for match in COMPILED_REGEX.finditer(text, re.UNICODE):
        token = match.group()
        if len(token) > 3:
            features.append(token)
    return features









def main():
    pass

if __name__ == "__main__":
    main()
