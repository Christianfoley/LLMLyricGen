# Lyre-LM: An Exploration of Fine-tuning Large Language Models for Lyric Generation
### Official implementation of Lyre-LM. To view the full exploration report please see [LyreReport.pdf](LyreReport.pdf).

## Overall Environment
**Make sure to have git lfs installed: instructions found here: https://git-lfs.com/**

Most python packages needed are in the ```requirements.txt``` file.

We recommend you make a new conda environment and run:
```
pip install -r requirements.txt
```

## Model Output Explorer

We provided a jupyter notebook that lets to expore some of the model outputs.  It is important to note that the songs in the dataset as well as the model outputs can contain mature or offensive content.  View discretion is advised.

Check run the files in [explore_generations.ipynb](explore_generations.ipynb) to see what out model outputted on the test set.  You can also compare to what llama outputs without finetuning.

## Embedding Instructions

Embeddings were generated using the Sentence Bert `all-mpnet-base-v2` model.

To view the embedding graphical results, run through the jupyter notebook [run_evaluation_metrics.ipynb](run_evaluation_metrics.ipynb)

## Musicality Evaluation Instructions

We provided a jupyter notebook to explore evaluation metrics (as well as view alternative embeddings!).

To explore the evaluation metrics, take a look at [run_evaluation_metrics.ipynb](run_evaluation_metrics.ipynb)

## Data visualization

To generate the rest of the figures from our report, as well as some additional visualizations of our data, please see [visualization_notebooks](visualization_notebooks).

## Interactive Demo
(Large file warning, this will download the model ~25 GB)

7 models are uploaded on to our Huggingface space: https://huggingface.co/cs182project

Inference should work on all of them.

To play with out model, follows these steps:

1. Make a new conda environment.

2. run ```pip3 install "fschat[model_worker,webui]```

3. run ```python3 -m fastchat.serve.cli --model-path cs182project/Llama-2-7b-chat-Lyre-LM-chat-checkpoint-100```

    You can pick any model from the Huggingface space, just change out the ```--model-path``` argument.

4. Have fun!

## Finetuning Instructions

IN ORDER TO FINETUNE THE LLAMA 7B MODEL, YOU NEED ~160GB OF VRAM.  ALL FINETUNING WAS DONE ON 2X A100 80GB

To finetune use [FastChat](https://github.com/lm-sys/FastChat/tree/main)

Instructions are as follows:

0. Request llama access via https://ai.meta.com/resources/models-and-libraries/llama-downloads/
    Approval is usually very fast if you use your berkeley.edu email.

    Make sure you have a Huggingface account, you need to prove to them you have llama access

    See Huggingface CLI documentation for logging in.

1. Create new Python environment, we recommend conda.

2. Run the following commands:
    ```
    git clone https://github.com/lm-sys/FastChat.git
    cd FastChat
    pip3 install --upgrade pip  # enable PEP 660 support
    pip3 install -e ".[model_worker,webui]"
    pip3 install -e ".[train]
    ```

3. Run the training command (make sure to update the ```--data_path``` to the actual path to the datafile found at LLMLyricGen/data/prompts/conversation_style_new_prompts.json found [here](data/prompts/conversation_style_new_prompts.json) with respect to your FastChat download location).  If you want to train from Llama-2-7B instead of the chat version, use ```--model_name_or_path meta-llama/Llama-2-7b-hf``` .

    ```
    torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
        --model_name_or_path  meta-llama/Llama-2-7b-chat-hf\
        --data_path ~/LLMLyricGen/data/prompts/conversation_style_new_prompts.json \
        --bf16 True \
        --output_dir output_llama-2-7b-chat-hf-lyre \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True
    ```

    Pick a batch size and nproc_per_node that is compatible with your setup.

4. Watch the loss go down! Fun!

5. If you want to try LoRA fine-tuning, tru this command:
    Use the ```--include localhost:0``` tag to pick the GPUs you want to run on.  Note that LoRA still takes around 40-80GB to run.  You can change the rank by changing ```--lora_r 256```.

    Note that to get this working, you might need to install some extra Python packages that FastChat does not naturally come with. As such, ensure that you have PEFT installed.  You can install it with ```pip install peft```.  You may also need to run ```pip install deepspeed```.

    Again, make sure the file path for the data is properly referencing [this file](data/prompts/conversation_style_new_prompts.json).

    ```
    deepspeed --include localhost:0 fastchat/train/train_lora.py \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
        --lora_r 256 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --data_path ~/LLMLyricGen/data/prompts/conversation_style_new_prompts.json \
        --bf16 True \
        --output_dir output_llama-2-7b-chat-lyre-chat_lora_rank256 \
        --num_train_epochs 6 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 12 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --q_lora True \
        --deepspeed playground/deepspeed_config_s2.json
    ```
## Prompt Generation

We used GPT-4-Turbo to generate prompts, but you can use any of OpenAI's models that are compatible with the chat completion API to run it. [Here is the file](generate_prompts/generate_prompts.py) for generating prompts.  You can see the prompts we used on line 11 and 12.

The arguments the run this are in the bottom of the script where the argparse arguments are defined.  This script is only compatible with [a csv schema like this file](SongLyricsScraper/1000_songs_no_annotation.csv).

## MT-Bench Radar Plots

To generate the radar plots, we copy code from [this colab notebook by Lmsys](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO).

We provide a customized copy in this repo, run [generate_mt_bench_plots.ipynb](mt_bench/generate_mt_bench_plots.ipynb) to replicate the plots.

## MT-Bench Results

MT-Bench is run using [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

To evaluate our model on MT-Bench do the following setup in you favorite python environment management system:

1. Run these commands:
    ```
    git clone https://github.com/lm-sys/FastChat.git
    cd FastChat
    pip install -e ".[model_worker,llm_judge]"
    ```
2. Run this to generate the outputs (to make this run in a reasonable time window, use a GPU)

    Pick a model to evaluate: [```cs182project/Llama-2-7b-chat-Lyre-LM-chat```, ```cs182project/Llama-2-7b-chat-Lyre-LM-chat-checkpoint-50```]
    ```
    python gen_model_answer.py --model-path cs182project/Llama-2-7b-chat-Lyre-LM-chat --model-id lyre-chat-small-batch-checkpoint-150
    ```
    if using  ```cs182project/Llama-2-7b-chat-Lyre-LM-chat-checkpoint-50``` change ```--model-id``` to ```lyre-chat-small-batch-checkpoint-50```

    Alternatively, you may copy our provided MT-Bench outputs found at ```mt_bench/lyre-chat-small-batch-checkpoint-150.jsonl``` or  ```mt_bench/lyre-chat-small-batch-checkpoint-150.jsonl``` into ```FastChat/fastchat/llm_judge/data/mt_bench/model_answers/``` or wherever you cloned FastChat.  You may need to make the model_answer directory.


3. Once inference is complete run this to generate judgements, you need an OpenAI API Key:

    ```
    python gen_judgment.py --model-list lyre-chat-small-batch-checkpoint-150
    ```
    and/or
    ```
    python gen_judgment.py --model-list lyre-chat-small-batch-checkpoint-50
    ```
5. Once judgement is done, run ```python show_result.py --model-list lyre-chat-small-batch-checkpoint-50 lyre-chat-small-batch-checkpoint-150```

## Training Curves, Hyper-Parameters, and Ablations

To replicate the training curves in the paper, run through [this ipynb](training_curves/create_training_visualizations.ipynb).  The batching graphs were pulled from WandB, so the files are directly included in the same directory as the notebook.
