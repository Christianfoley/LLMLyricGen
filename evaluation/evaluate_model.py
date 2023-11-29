from score_acculumator import ScoreAccumulator
import argparse, json, os


def main(filepath, measures, writeback=False, outpath=""):
    """
    Evaluate a model using specified measures given a filepath to database of songs.

    Database must be formatted as follows:
    [
        #dict 1 for song 1
        {
            "id": (int) ...,
            "prompt": (str) ...,                  #optional
            "model_response": (str) ...,
            "target_response": (str) ...          #optional
        },
        #dict 2 for song 2
        {
            " "
        },
        .
        .
        .
    ]

    Parameters
    ----------
    filepath : str
        path to database .json file
    measures : list
        list of measures to evaluate model outputs, from {'diversity','meter','syllable'}
    writeback : bool, optional
        Whether to write evaluation scores to filepath or to output, by default False
    outpath : str, optional
        path to output .json file if writeback is False, by default ""

    Raises
    ------
    FileNotFoundError
    """
    # read file
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            database = json.load(f)
    else:
        raise FileNotFoundError(f"No such file exists: {filepath}")

    if not writeback:
        if not os.path.exists(outpath):
            raise FileNotFoundError(f"No such file exists: {outpath}")
    else:
        outpath = filepath

    # evaluate for measures
    accumulator = ScoreAccumulator(
        measures=measures,
        require_prompt="prompt" in database[0],
        require_target="target_response" in database[0],
    )
    accumulator.score_all_songs(database)

    # print total scores
    for measure in measures:
        pred_score = accumulator.get_total_pred_score(measure)
        target_score = accumulator.get_total_target_score(measure)
        print(f"Score: pred {pred_score:2f}, target {target_score:2f} : ({measure})")

    # save evaluation
    out_database = []
    for id, song_dict in accumulator._database.items():
        song_dict["id"] = int(id)
        out_database.append(song_dict)

    with open(filepath, "w") as f:
        f.write(json.dumps(out_database, indent=2, separators=[",", ":"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model given a filepath to a database of songs."
    )
    parser.add_argument(
        "filepath", type=str, help="The path to the file to be processed"
    )
    parser.add_argument(
        "--measures",
        default=["diversity", "meter", "syllable", "phonetic"],
        nargs="+",
        help="List of measures to evaluate. From {'diversity','meter','syllable','phonetic'}",
    )
    parser.add_argument(
        "--writeback",
        type=bool,
        default=True,
        help="Write evaluation scores back to the same dict or create a new one",
    )
    parser.add_argument(
        "--output",
        default="",
        type=str,
        help="The path to the write output scores if writeback is false",
    )

    args = parser.parse_args()
    main(args.filepath, args.measures, args.writeback, args.output)
