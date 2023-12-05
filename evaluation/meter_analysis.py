import nltk
import re
import string

import evaluation.syllable_analysis as sylco

nltk.download("cmudict")
WORDS = nltk.corpus.cmudict.dict()


def get_stress(word):
    syllables = WORDS.get(word.lower())
    if syllables:
        stresses = "".join(
            [phoneme[-1] for phoneme in syllables[0] if phoneme[-1].isdigit()]
        )
        return stresses
    else:
        return "Word not found in CMU Pronouncing Dictionary"


def get_line_meter(line):
    """Credit to https://stackoverflow.com/questions/9666838/"""

    # Clean punctuation
    exclude = set(string.punctuation)
    exclude.remove("-")
    line = "".join(ch for ch in line if ch not in exclude)

    # stem words
    words = re.split(r"[\s-]+", line)
    stemmer = nltk.stem.SnowballStemmer("english")
    stemmed_words = []
    for word in words:
        if word[-3:] == "ing" or word[-2:] == "ly":
            stemmed_words.append(word)
        else:
            stemmed_words.append(stemmer.stem(word))

    # get meter
    line_stress = []
    for word in stemmed_words:
        if word not in WORDS:
            line_stress.append("0" * sylco.count_syllables(word))
        else:
            line_stress.append(get_stress(word))

    return map(str, line_stress)
