from collections import defaultdict, Counter
import copy, pprint, re
import itertools
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

import evaluation.scoring_metrics as metrics
import evaluation.text_processing_utils as text_utils


class ScoreAccumulator(object):
    def __init__(
        self,
        measures=["diversity"],
        default_matching="",
        require_prompt=False,
        require_target=True,
        verbose=True,
    ):
        """
        Score accumulation object for measuring model inference performance
        on a dataset of songs.

        Capable of measuring:
            Lexical diversity : "diversity"
            Meter consistency : "meter"
            Syllabic consistency : "syllable"

        Parameters
        ----------
        measures : list, optional
            list of measures to calculate for each song, by default ["diversity"]
        default_matching : str
            string defining the default stanza or line matching for comparison
        require_prompt : bool, optional
            whether to require the model prompt for each song, by default False
        require_target : bool, optional
            whether to requre a target response for each song, by default True
        """
        self.measures = measures
        self.default_matching = default_matching
        self.require_prompt = require_prompt
        self.require_target = require_target
        self.verbose = verbose

        self.__total_score = Counter({measure: 0 for measure in self.measures})
        self.__total_target_score = Counter({measure: 0 for measure in self.measures})
        self.__database = {}

        # Safety checks on measures
        supported_meas = {
            "diversity",
            "meter",
            "syllable",
            "semantics",
            "semantics_internal",
            "string_similarity",
            "head_similarity",
            "tail_similarity",
        }
        self.internal_measures = {"diversity", "semantics_internal"}
        for measure in self.measures:
            if measure not in supported_meas:
                raise NotImplementedError(f"Only measures {supported_meas} supported.")

        # initiate measure functions
        self.semantic_model = None
        if "semantics" in measures or "semantics_internal" in measures:
            assert torch.cuda.is_available(), "Semantic measures require a gpu"
            # use "all-mpnet-base-v2" for slightly better & much slower embeddings
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.measure_functions = {
            "diversity": metrics.measure_lex_div,
            "meter": metrics.measure_meter,
            "syllable": metrics.measure_syllable,
<<<<<<< HEAD
            "semantics": lambda p1, p2: metrics.measure_compared_semantics(
                self.semantic_model, p1, p2
            ),
            "semantics_internal": lambda p1: metrics.measure_internal_semantics(
                self.semantic_model, p1
            ),
            "string_similarity": metrics.measure_similarity,
            "head_similarity": lambda p1, p2: metrics.measure_similarity(p1, p2, 2),
            "tail_similarity": lambda p1, p2: metrics.measure_similarity(p1, p2, -2),
        }

=======
            "phonetic": metrics.measure_phonetic_similarity,
        }

        supported_meas = {"diversity", "meter", "syllable", "phonetic"}
        for measure in self.measures:
            if measure not in {"diversity", "meter", "syllable", "phonetic"}:
                raise NotImplementedError(f"Only measures {supported_meas} supported.")

>>>>>>> 3-evaluation-metric-4
    def get_total_pred_score(self, measure):
        """
        Return current score totals of model predictions for a given
        measure

        Parameters
        ----------
        measure : str
            measure to get score for

        Returns
        -------
        float
            average (by song) cumulative score
        """
        measure_score = self.__total_score[measure]
        return measure_score / len(self.__database)

    def get_total_target_score(self, measure):
        """
        Return current score total of target responses for a given
        measure

        Parameters
        ----------
        measure : str
            measure to get score for

        Returns
        -------
        float
            average (by song) cumulative score
        """
        measure_score = self.__total_target_score[measure]
        return measure_score / len(self.__database)

    def get_database(self):
        return self.__database

    def get_ids(self):
        return list(self.__database.keys())

    def get_prompts(self, ids=[]):
        if ids:
            return [self.__database[id]["prompt"] for id in ids]
        return [self.__database[songid]["prompt"] for songid in self.get_ids()]

    def get_preds(self, ids=[]):
        if ids:
            return [self.__database[id]["model_response"] for id in ids]
        return [self.__database[songid]["model_response"] for songid in self.get_ids()]

    def get_targets(self, ids=[]):
        if ids:
            return [self.__database[id]["target_response"] for id in ids]
        return [self.__database[songid]["target_response"] for songid in self.get_ids()]

    def get_pred_scores(self, ids=[]):
        if ids:
            return [self.__database[id]["pred_scores"] for id in ids]
        return [self.__database[songid]["pred_scores"] for songid in self.get_ids()]

    def get_target_scores(self, ids=[]):
        if ids:
            return [self.__database[id]["target_scores"] for id in ids]
        return [self.__database[songid]["target_scores"] for songid in self.get_ids()]

    def score_all_songs(self, song_database):
        """
        Scores all songs in a song database (a list of song dictionaries) and adds
        them to accumulators database.

        Parameters
        ----------
        song_database : list
            list of song dictionaries containing {id, prompt, model_response, and
            target_response} keys
        """
        # TODO: add safety checks

        for song_dict in tqdm(song_database, position=0):
            try:
                self.score_song(song_dict)
            except Exception as e:
                print(f"Error processing song: {song_dict['id']} {e}")

    def score_song(self, song_dict):
        """
        Adds song to accumulator and scores song.
        Song dicts are formatted as follows:
        {
            id: unique id number,
            prompt: prompt string,                      #if required
            model_response: response from model,
            target_response: ground truth text          #if required
        }

        Parameters
        ----------
        song_dict : dict
            dict containing model predictions and targets for song
        """
        song_dict = copy.deepcopy(song_dict)
        try:
            id = song_dict["id"]
            pred = song_dict["model_response"]
            if self.require_prompt:
                song_dict["prompt"]
            if self.require_target:
                target = song_dict["target_response"]
        except Exception as e:
            raise AssertionError(f"Provided song dict formatted incorrectly: {e}")
        assert id not in self.__database, f"Multiple entries found for unique id: {id}"

        # If model refused due to value alignment just pass:
        if "I apologize, but I cannot fulfill your request" in pred:
            if self.verbose:
                print(f"Model refused song {id} for value alignment reasons.")
            return

        # Song quality evaluations (via internal similarity and structure)
        scores = {
            "pred_scores": {measure: 0 for measure in self.measures},
            "target_scores": {measure: 0 for measure in self.measures},
        }
        pred_stanzas, pred_matching = self.get_stanzas(pred, get_matching=True)
        target_stanzas, target_matching = self.get_stanzas(target, get_matching=True)
        for measure in (
            tqdm(self.measures, position=1, leave=0) if self.verbose else self.measures
        ):
            scores["pred_scores"][measure] = self.measure_stanzas(
                pred_stanzas,
                self.measure_functions[measure],
                matching="" if measure in self.internal_measures else pred_matching,
            )
            scores["target_scores"][measure] = self.measure_stanzas(
                target_stanzas,
                self.measure_functions[measure],
                matching="" if measure in self.internal_measures else target_matching,
            )

        self.__total_score += Counter(scores["pred_scores"])
        self.__total_target_score += Counter(scores["target_scores"])

        song_dict.pop("id")
        song_dict.update(scores)
        self.__database[id] = song_dict

    def get_stanzas(self, song_text, get_matching=False):
        """
        Given the raw text of a song, gets the stanza paragraphs according to
        their in-text label as {hook, verse, chorus, bridge, intro, outro}

        Parameters
        ----------
        song_text : str
            raw song text
        get_matching : bool, optional,
            whether to return a matching

        Returns
        -------
        list : str
            raw text, stripped of their stanza identifiers, split into stanzas
        """
        stanza_pairs = text_utils.get_stanzas(song_text)

        matching_map = {}
        matching_str = ""
        match_idx = 0  # NOTE: using ints prohibits > 10 keywords per song

        stanzas = []
        for i, (kword, stanza) in enumerate(stanza_pairs):
            assert match_idx < 10, f"Stanza keywords are too complex, please simplify"
            stanzas.append(
                [n for n in stanza.split("\n") if len(re.sub(r"\s+", "", n)) > 0]
            )

            if i == 0:
                matching_map[kword] = str(match_idx)
                matching_str = str(match_idx)
                match_idx += 1
            elif kword in matching_map:
                matching_str += "-" + matching_map[kword]
            else:
                matching_map[kword] = str(match_idx)
                matching_str += "-" + matching_map[kword]
                match_idx += 1

        if get_matching:
            return stanzas, matching_str
        else:
            return stanzas

    def build_matches(self, stanzas, matching=""):
        """
        Builds matchings between stanzas given a matching string. If no matching
        string provided, matchings are just the stanzas themselves.

        Parameters
        ----------
        stanzas : list
            list of input strings (stanzas) for different stanzas in a song
        matching : str, optional
            string defining the stanza or line matching for comparison, by default ""

        Returns
        -------
        list
            list of "matched" stanza pairings
        """
        if matching == "":
            return [(s,) for s in stanzas]
        matching = matching.split("-")
        assert len(matching) == len(stanzas), "Incorrect matching to stanzas"

        matches_dict = defaultdict(lambda: [])
        pairings = []

        # map stanzas to their matching char
        for i, p in enumerate(stanzas):
            matches_dict[matching[i]].append(p)

        # build pairings
        for matches in matches_dict.values():
            for p1, p2 in itertools.combinations(matches, 2):
                pairings.append([p1, p2])

        return pairings

    def measure_stanzas(self, stanzas, measure_function, matching=""):
        """
        Computes and returns a measure of a list of stanzas (string segments made up of numerous
        lines). If measure is a consistency score, a comparison stanza is required.

        Comparison metrics will be computed along the cartesian product of stanzas with matching
        keys.

        For example, given:
            stanzas = ["line1", "line2", "line3", "line4"]
            matching = "a-b-a-b"
                -> Comparisons computed between [("line1", "line3"), ("line2", "line4")]

        Parameters
        ----------
        stanzas : list
            list of input strings (stanzas) for different stanzas in a song
        measure_function : function
            callable function that takes in a pairing and returns a float
        matching : str
            string defining the stanza or line matching for comparison

        Returns
        -------
        float
            averaged (across measurings) measure score
        """
        pairings = self.build_matches(stanzas, matching)

        if len(pairings) == 0:
            # NOTE: dangerous, since if no pairings the song structure could be fine..
            return 0

        score = 0

        for pairing in pairings:
            score += measure_function(*pairing)

        final_score = score / len(pairings)

        return final_score
