import re
import math
import unidecode
import copy
from pprint import pp

from lexical_diversity import lex_div as ld
from dtaidistance import dtw
from fastdtw import fastdtw
import numpy as np

import encode_lines as encode


def measure_lex_div(p1, mode="mtld"):
    """
    Measure of lexical diversity of a set of lines

    Parameters
    ----------
    p1 : list(str)
        list of lines of paragraph or song as strings
    mode : str, optional
        lexical diversity metric, by default "mtld"

    Returns
    -------
    float
        lexical diversity score
    """
    lines = " ".join(p1)
    flem_tokens = ld.flemmatize(unidecode.unidecode(lines))
    if mode == "avg_ttr":
        lex_div = ld.ttr(flem_tokens) / len(flem_tokens)
    elif mode == "mtld":
        lex_div = ld.mtld(flem_tokens)
    elif mode == "avg_mtld":
        lex_div = ld.mtld_ma_wrap(flem_tokens)
    return lex_div


def measure_meter(p1, p2):
    """
    Measure: meter consistency

    Meter consistency between two lines or paragraphs is defined as the negative
    exponential of the edit distance between their stress encodings.

    Note that this mapping limits the "maximal" error, as extremely high syllable differences
    will incur exponentially lower loss. The score is scaled between 0 and 100

    Parameters
    ----------
    p1 : list
        paragraph as a list of line strings
    p2 : list
        comparison paragraph as a list of line strings

    Returns
    -------
    float
        score of meter consistency
    """
    # encode into syllabic stress indicators
    p1 = [encode.encode_line_meter_count(line) for line in p1]
    p2 = [encode.encode_line_meter_count(line) for line in p2]

    p1_string = "".join(line_stress for line_stress in p1).replace(" ", "").strip()
    p2_string = "".join(line_stress for line_stress in p2).replace(" ", "").strip()

    p1_string = re.sub(r"\s+", "", p1_string, flags=re.UNICODE)
    p2_string = re.sub(r"\s+", "", p2_string, flags=re.UNICODE)

    edit_dist = levenshteinDistance(p1_string, p2_string)
    return edit_dist  # min(100, 100 / (edit_dist + 2e-7))  # math.exp(-0.1 * avg_edit_distance)


def measure_syllable(p1: list, p2: list):
    """
    Measure: syllable consistency

    We perform an altered version of the syllable DTW described in the methods
    section of https://staff.aist.go.jp/m.goto/PAPER/TIEICE202309watanabe.pdf.
    By recognizing that sometimes entire lines are out of "sync" between paragraphs,
    we first minize the DTW via sliding window between the two paragraphs, then compute
    the DTW line-by-line at the minimized offset.

    Syllable consistency between two lines or paragraphs is defined as the
    negative exponential of the DTW distance between two paragraph syllables.

    Note that this mapping limits the "maximal" error, as extremely high syllable differences
    will incur exponentially lower loss. The score is scaled between 0 and 100

    Parameters
    ----------
    p1 : list
        paragraph as a list of line strings
    p2 : list
        comparison paragraph as a list of line strings

    Returns
    -------
    float
        length-normalized syllabic consistency score
    """
    score = 0

    # encode p1 and p2 into syllable counts
    enc_fn = encode.encode_line_syllable_count

    def encode_paragraph(par):
        out = []
        for line in par:
            encoded_line = enc_fn(line).split(" ")
            encoded_line = [word for word in encoded_line if word]
            if len(encoded_line) == 0:
                out.append([0])
            else:
                out.append(list(map(int, encoded_line)))
        return out

    p1, p2 = encode_paragraph(p1), encode_paragraph(p2)

    # find and shift/crop p1 and p2 to the best matching offset
    _, p1_c, p2_c = min_dtw_offset(p1, p2, return_cropped=True, use_short_window=True)

    for i in range(len(p1_c)):
        score += abs(sum(p1_c[i]) - sum(p2_c[i]))

    return score  # min(100, 100 / (score + 2e-7))  # math.exp(-0.1 * score)

def measure_phonetic_similarity(p1, p2):
    """
    Measure: phonetic similarity

    Phonetic similarity is calculated through a a normalized Levenshtein edit distance
    between the phonetic transcriptions of two lyric lines. We use the IPA form
    of the word to represent its pronunciation.

    Parameters
    ----------
    p1 : list
        paragraph as a list of line strings
    p2 : list
        comparison paragraph as a list of line strings

    Returns
    -------
    float
        phonetic edit distance
    """

    # encode into IPA representations aka pronunciations
    p1 = [encode.encode_line_pronunciation(line) for line in p1]
    p2 = [encode.encode_line_pronunciation(line) for line in p2]

    p1_string = "".join(line_pronunciation for line_pronunciation in p1).replace(" ", "").strip()
    p2_string = "".join(line_pronunciation for line_pronunciation in p2).replace(" ", "").strip()

    p1_string = re.sub(r"\s+", "", p1_string, flags=re.UNICODE)
    p2_string = re.sub(r"\s+", "", p2_string, flags=re.UNICODE)

    p1_string = re.sub(r'[^\w]', "", p1_string, flags=re.UNICODE)
    p2_string = re.sub(r'[^\w]', "", p2_string, flags=re.UNICODE)

    edit_dist = levenshteinDistance(p1_string, p2_string)
    return edit_dist


def min_dtw_offset(p1, p2, return_cropped=True, use_short_window=True):
    """
    Use a sliding window (of lines) to find the line index offset which minimizes
    the syllabic DTW between two paragraphs.

    In addition to outside sliding window, setting inner_window > 0 allows for
    an moving inner window across the smaller signal to find a cropping

    Parameters
    ----------
    p1 : list
        list of lists of song lines encoded as syllable counts
    p2 : list
        list of song lines encoded as syllable counts
    return_cropped : bool, optional
        whether to return the cropped min window for p1 and p2 ro the offset, by
        default True
    use_short_window : bool, optional
        whether to compare at smaller or larger paragraphs length, by default true
        Note: if False, p1 and p2 will not be the same length

    Returns
    -------
    int
        dtw minimizing offset value of smaller signal in larger signal
    list
        (p1) cropped or uncropped p1, squared with zeros
    list
        (p2) cropped or uncropped p2, squared with zeros
    """
    switched = False
    if len(p1) < len(p2):
        switched = True
        p1, p2 = p2, p1

    # crop or pad to same number of lines
    win_length = len(p2) if use_short_window else len(p1)

    # square by padding line lengths with zeros
    sig1, sig2 = copy.deepcopy(p1), copy.deepcopy(p2)
    max_val = max(max(len(l) for l in sig1), max(len(l) for l in sig2))
    for i in range(len(p1)):
        sig1[i] += [0] * (max_val - len(sig1[i]))
    for i in range(len(p2)):
        sig2[i] += [0] * (max_val - len(sig2[i]))
    sig1, sig2 = np.sum(np.array(sig1), axis=1), np.sum(np.array(sig2), axis=1)

    # compute dtw with a sliding window to find the best offset
    sig1, sig2 = np.pad(sig1, (win_length, win_length)), sig2.astype(np.double)
    min_idx, min_error = 0, np.inf
    for j in range(sig1.shape[0] - win_length):
        sig1_win = sig1[j : j + sig1.shape[0]].astype(np.double)

        error = dtw.distance_fast(sig1_win, sig2, inner_dist="euclidean")

        if error < min_error:
            min_error = error
            min_idx = j

    # crop to window
    if return_cropped:
        p1 = [[0]] * win_length + p1 + [[0]] * win_length
        p1 = p1[min_idx : min_idx + win_length]

    if switched:
        p1, p2 = p2, p1

    return min_idx - win_length, p1, p2


def levenshteinDistance(s1, s2):
    """
    DP edit distance implementation
    credit to https://stackoverflow.com/questions/2460177

    Parameters
    ----------
    s1 : str
        first string to compare
    s2 : str
        second string to compare

    Returns
    -------
    int
        edit distance between two strings (absolute)
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]
