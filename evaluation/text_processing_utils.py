import re
from unidecode import unidecode

STANZA_KEYWORDS = {
    "pre-hook",
    "post-hook",
    "pre-drop",
    "pre-chorus",
    "post-chorus",
    "pre-coro",
    "post-coro",
    "breakdown",
    "drop",
    "hook",
    "verse",
    "chorus",
    "bridge",
    "intro",
    "outro",
    "refrain",
    "guitar solo",
    "solo",
    "letra de",
    "instrumental",
    "verso",
    "coro",
    "couplet",
    "pont",
    "ponte",
    "interlude",
    "part",
    "refrão",
    "refrao",
}


def get_kword(delin):
    """Gets kword readable string from matched delineator"""
    delin = delin.split(":")[0]
    delin = re.sub(r"\d+", "", delin)
    return delin.strip()


def clean_song(text):
    """
    Custom rules for "cleaning" the song data to disambiguate stanza
    delineaters

    Parameters
    ----------
    text : str
        raw song data

    Returns
    -------
    str
        cleaned song data
    """
    text = unidecode(text).lower()

    # Replace all "[?]", "[chuckles]", "[laughs]", "[Mumbling]" with "nan"
    text = re.sub(r"\[\?\]|\[chuckles\]|\[laughs\]|\[Mumbling\]", "nan", text)
    text = re.sub(r"\(\?\)|\(chuckles\)|\(laughs\)|\(Mumbling\)", "nan", text)

    # Replace all "]:" with "]\n"
    text = re.sub(r"\]:", "]\n", text)
    text = re.sub(r"\):", ")\n", text)

    # Replace all "[X]" with "nan" where X is any number of "." characters
    text = re.sub(r"\[\.*?\]", "nan", text)
    text = re.sub(r"\(\.*?\)", "nan", text)

    # For any remaining bracketed texts replace with kword readable string and add a newline

    def match_to_kword_group(match):
        kword = get_kword(match.group(2))

        for keyword in STANZA_KEYWORDS:
            if kword.startswith(keyword):
                return kword
        return False

    def replace_bracketed_if_match(match):
        kword = match_to_kword_group(match)
        if kword:
            return f"[{kword}]\n"
        else:
            return match.group()

    text = re.sub(r"(\[|\()([\s\S]*?)(\]|\))", replace_bracketed_if_match, text)

    # Finally remove any pre/post-ambles:
    # 1.) the songtext should begin with a successful match (matches to a keyword)
    # 2.) the should should end with the LAST \n\n that is Not followed directly by a match
    first_delin_idx = 0
    delins = list(re.finditer(r"(\[)([\s\S]*?)(\])", text))
    for match in delins:
        if match_to_kword_group(match):
            first_delin_idx = match.start()
            break
    text = text[first_delin_idx:]

    trailing_double_nls = list(re.finditer(r"(?<!\])\n\n(?!(\[)([\s\S]*?)(\]))", text))
    for match in trailing_double_nls:
        if match.start() >= first_delin_idx:
            text = text[: match.start()]
            break

    return text


def get_stanzas(text):
    """
    Process song as raw text to return a list of stanza - keyword pairings.
    If keyword match is unidentified, pairs with entire match rather than just the
    known keyword

    Parameters
    ----------
    text : str
        raw song text

    Returns
    -------
    list(tuple)
        list of tuple (keyword, stanza_text) pairings
    """
    stanzas = []
    text = clean_song(text)

    # Find all identifiers inside brackets
    matches = re.findall(r"\[([\s\S]*?)\]", text)
    split_text = re.split(r"\[(?:[\s\S]*?)\]", text)[1:]

    # pair text in stanzas with existing keyword or new match
    for i, match in enumerate(matches):
        matched_with_kword = False

        for keyword in STANZA_KEYWORDS:
            if match.startswith(keyword):
                stanzas.append((keyword, split_text[i]))
                matched_with_kword = True
                break

        if not matched_with_kword:
            stanzas.append((match, split_text[i]))

    # remove empty stanzas
    stanzas = [(keyword, stanza) for keyword, stanza in stanzas if stanza.strip()]

    return stanzas


def find_surrounding_chars(text, pattern, before=50, after=50):
    """Helpful testing utility"""
    regex_pattern = f".{{0,{before}}}{pattern}.{{0,{after}}}"
    return re.findall(regex_pattern, text)
