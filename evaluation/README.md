# Evaluation
This module contains utilities and scripts for empirical, numerical evaluation of model outputs. Much of the code written in this module is guided by methodologies described in [this paper](https://staff.aist.go.jp/m.goto/PAPER/TIEICE202309watanabe.pdf) (Watanabe, 2023). 

### Data formatting
Using the [evaluate_model.py](evaluate_model.py) script, evaluation can be run on an existing `json` database of outputs. Database must contain a list of song containers in the following format:
```json
[
    {
        "id": 0,
        "prompt": "some prompt here",
        "model_response": "some response here",
        "target_response": "some ground truth here"
    },
    {
        "id": 1,
        "prompt": "some prompt here",
        "model_response": "some response here",
        "target_response": "some ground truth here"
    },
    .
    .
    .
]
```

### Quick start
Run the following script to quick start an evaluation:
```bash
py evaluation/evaluate_model.py <path_to_your_database> 
```

To run different measures, consider passing in a list of measures into the `--measures` argument:
```bash
py evaluation/evaluate_model.py <path_to_your_database> --measures diversity meter syllable
```