import syllable_analysis as sylco
import meter_analysis as metco
import re, unidecode


def prep_encoding(text):
    """
    Cleans text by removing whitespace, replacing nonstandard characters with
    their standard counterparts, etc

    Parameters
    ----------
    text : str
        any text

    Returns
    -------
    str
        cleaned text
    """
    text = unidecode.unidecode(text).strip()

    if not re.search(r"\S", text):
        text = ""
    return text


def encode_line_meter_count(line, to_stdout=False):
    """
    Encodes a song line (line of text) into a line of digit words representing
    the stress of each word in the line.

    Ex:
        what so proudly we hailed at the twilight's last gleaming
            -> 1 1 10 1 1 1 0 12 1 10

    Parameters
    ----------
    line : str
        string of words (line)
    to_stdout : bool, optional
        whether to print to stdout, by default False

    Returns
    -------
    str
        string of stress encodings (digits)
    """
    line = prep_encoding(line)

    if line == "":
        if to_stdout:
            print(line)
        return ""

    line_stress_list = metco.get_line_meter(line)
    out = " ".join(line_stress_list)

    if to_stdout:
        print(out)
    return out


def encode_line_syllable_count(line, to_stdout=False):
    """
    Encodes a song line (line of text) into a line of digits representing
    the number of syllables per line.
    Ex:
        the quick brown fox jumps over the lazy dog
            -> 1 1 1 1 1 1 1 2 1

    Parameters
    ----------
    line : str
        string of words (line)
    to_stdout : bool, optional
        whether to print to stdout, by default False

    Returns
    -------
    string
        string of digits, one digit per word
    """
    line = prep_encoding(line)

    if line == "":
        if to_stdout:
            print(line)
        return line

    words = re.findall(r"\b\w+\b", line)
    syllable_counts = [sylco.count_syllables(word) for word in words]

    out = " ".join(map(str, syllable_counts))

    if to_stdout:
        if len(syllable_counts) > 0:
            out += " " * (30 - len(out))
            out += f": {sum(syllable_counts)}"
        print(out)
    return out
