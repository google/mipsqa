# README

**This is not an official Google product**

This project contains code for training and running an extractive question answering model on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). All methods and models contained in this project are described in the [technical report](https://github.com/google/mipsqa/blob/master/mips-qa.pdf). Any extensions of this work should cite the report as:

```
@misc{SeoKwiatParikh:2017,
  title = {Question Answering with Maximum Inner Product Search},
  author = {Minjoon Seo and Tom Kwiatkowski and Ankur Parikh},
  url = {...},
}
```

# 0. Requirements and data
- Basic requirements: Python 2 or 3, wget (if using MacOS. You can also download yourself looking at the `download.sh` script)
- Python packages: tensorflow 1.3.0 or higher, nltk, tqdm
- Data: SQuAD, GloVe, nltk tokenizer

To install required packages, run:
```bash
pip install -r requirements.txt
```

To download data, run:
```bash
chmod +x download.sh; ./download.sh
```

Change the directories where the data is stored if needed, and use them for runs below.

# 1. Train and test (draft mode)
If you are using default directories for the data:
```bash
export SQUAD_DIR=$HOME/data/squad
export GLOVE_DIR=$HOME/data/glove
```

First, preprocess train data:
```bash
python squad_prepro_main.py --from_dir $SQUAD_DIR --to_dir prepro/draft/sort_filter --glove_dir $GLOVE_DIR --sort --filter --draft
```
Note the `--draft` flag, which only processes a portion of the data for fast sanity check. Make sure to remove this flag when doing real training and test.
`--filter` filters out very long examples, which can slow down training and cause memory issues.

Second, preprocess for test data, which does not filter any example:
```bash
python squad_prepro_main.py --from_dir $SQUAD_DIR --to_dir prepro/draft/sort_filter/sort --glove_dir $GLOVE_DIR --sort --draft --indexer_dir prepro/draft/sort_filter
```

Third, train a model:
```bash
python train_and_eval.py --output_dir /tmp/squad_ckpts --root_data_dir prepro/draft/sort_filter/ --glove_dir $GLOVE_DIR --oom_test
```
In general, `--oom_test` is a flag for testing if your GPU has enough memory for the model, but it can also serve as a quick test to make sure everything runs.

Fourth, test the model:
```bash
python train_and_eval.py --root_data_dir prepro/draft/sort_filter/sort --glove_dir $GLOVE_DIR --oom_test --infer --restore_dir /tmp/squad_ckpts
```
Note that `--output_dir` has to be changed to `--restore_dir`, and also `--infer` flag has been added.
Instead of using data from `prepro/draft/sort_filter/`, it is using `prepro/draft/sort_filter/sort`, which does not filter any example.
This outputs the json file in `--restore_dir` that is compatible with SQuAD official evaluator.
If you want to run this fully (no draft mode), remove `--draft` and `--oom_test` flags when applicable.
