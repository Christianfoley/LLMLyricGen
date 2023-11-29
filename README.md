# Lyre-LM: An Exploration of Fine-tuning Large Language Models for Lyric Generation

### Makes sure to have git lfs installed: instructions found here: https://git-lfs.com/

# Model Output Explorer

We provided a jupyter notebook that lets to expore some of the model outputs.  It is important to note that the songs in the dataset as well as the model outputs can contain mature or offensive content.  View discretion is advised.

Check run the files in ```explore_generations.ipynb``` to see what out model outputted on the test set.  You can also compare to what llama outputs without finetuning.

# Embedding Instructions

Embeddings were generated through embedding_generation/embed_model_response.py

To view the embedding graphical results, run through the jupyter notebook ```embedding_generation/evaluate_embedding_accuracy.ipynb```

# Musical Metrics Instructions

# Interactive Demo
(Large file warning, this will download the model ~25 GB)

To play with out model, follows these steps:

1. Make a new conda environment.

2. run ```pip3 install "fschat[model_worker,webui]```

3. run ``` python3 -m fastchat.serve.cli --model-path cs182project/llama-2-7b-chat-lyre-lm```

4. Have fun!

# Finetuning Instructions

IN ORDER TO FINETUNE THE LLAMA 7B MODEL, YOU NEED ~160GB OF VRAM.  ALL FINETUNING WAS DONE ON 2X A100 80GB

To finetune use [FastChat](https://github.com/lm-sys/FastChat/tree/main)

Instructions are as follows:

0. Request llama access via https://ai.meta.com/resources/models-and-libraries/llama-downloads/
    Approval is usually very fast if you use you berkeley.edu email.

    Make sure you have a Huggingface account, you need to prove to them you have llama access

    See Huggingface CLI documentation for logging in.

1. Create new Python environment, we recommend conda.

2. Run the following commands:
    ```
    git clone https://github.com/lm-sys/FastChat.git
    cd FastChat
    pip3 install --upgrade pip  # enable PEP 660 support
    pip3 install -e ".[model_worker,webui]"
    pip3 install -e ".[train]"
    ```

3. Run the training command:
    ```
    torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
        --data_path cs182project/182-final-project \
        --bf16 True \
        --output_dir output_llama-2-7b-chat-hf-lyre \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 101 \
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

