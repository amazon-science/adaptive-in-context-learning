# AdaICL: Which Examples to Annotate for In-Context Learning? Towards Effective and Efficient Selection

In this work, we investigate an active learning approach for ICL, where there is a limited budget for annotating examples. We propose a model-adaptive optimization-free algorithm, termed AdaICL, which identifies examples that the model is uncertain about, and performs semantic diversity-based example selection. Diversity-based sampling improves overall effectiveness, while uncertainty sampling improves budget efficiency and helps the LLM learn new information. Moreover, AdaICL poses its sampling strategy as a Maximum Coverage problem, that dynamically adapts based on the model’s feedback and can be approximately solved via greedy algorithms. 
![AdaICL algorithm.](assets/AdaICL_alg.pdf "AdaICL algorithm.")



## Installation
To establish the environment, run this code in the shell:
```
conda env create -f selective_annotation.yml
conda activate selective_annotation
```
We follow the general setup of Votek (<https://github.com/xlang-ai/icl-selective-annotation>).

## Usage

### Datasets

All datasets will be automatically downloaded from huggingface/datasets and stored here.

### End-to-end pipeline: selection, inference, evaluation
GPT-Neo as the in-context learning model, TREC and SST2 as the tasks, and AdaICL  as the selective annotation method, with additional budget of 20.
```
CUDA_VISIBLE_DEVICES=0 ./scripts/run_adaicl.sh
```

Example:
```
CUDA_VISIBLE_DEVICES=0 python main_adaptive_phases.py --evaluate_calibration --few_shot 5 --task_name ag_news --selective_annotation_method "ada_icl_default" --model_cache_dir "models" --data_cache_dir "datasets" --output_dir outputs --annotation_size 20 --model_name "gpt-neo" --seed 0 --init "cluster"  --sample_k 
```


## Directory Layout
Below you can find the scripts to reproduce the key results.

```bash
./active-in-context-learning
|---- MetaICL/                      # the model will be loaded similar to MetaICL for classification problems. That way we do not encouter invalid label generation.
|---- logs/                         # Folder for storing logfiles.
|---- outputs/                      # Folder for storing output results.
|---- scripts/                      # Run these scripts to reproduce results.
|
|---- algorithms.py                 # k-means, fast-votek, model_uncertainty_estimation, votek utilies
|---- annotation_methods.py         # Supported active learning algos.
|---- get_task.py                   # Dataset-specific utilies.
|---- main_adaptive_phases.py       # Execution of AL algos in an adaptive manner (inductive).
|---- main_generative.py            # Generation tasks.
|---- prompt_retrieval.py           # Retrieve prompts from annotated pool.
|---- utils.py                      # BERT embeddings, plots, calibration error etc.
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.