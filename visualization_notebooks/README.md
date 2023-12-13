## Additional data & figure visualization notebooks

- [profanity.ipynb](profanity.ipynb): An analysis of profanity usage statistics between different models. We find that further finetuning increases profanity usage, likely due to model forgetting of value alignment.
- [human_feedback.ipynb](human_feedback.ipynb): An analysis of our human feedback surveys. We find that humans vastly prefer our model outputs for rap, and are even for pop.
- [swift_LM.ipynb](swift_LM.ipynb): A data analysis & visualization of ground-truth n-gram frequency between baseline and Taylor Swift finetuned models. We find that lyre-swift (the model finetuned from lyre) tends to perform less plagarism.
- [create_training_visualizations](create_training_visualizations.ipynb): Analysis notebooks for isualizing data from finetuning ablations and monitoring.
- [generate_mt_bench_plots.ipynb](generate_mt_bench_plots.ipynb): Analysis of task-specific catastrophic forgetting; figure generation from mt-bench.