# Dataset Preprocess Guide

In this document, we demonstrate how to preprocess the datasets for running the experiments in the paper. For each dataset, we start from downloading it, and then preprocess it to the format that can be used by the LongServe system, and finally check the distribution of the dataset.

It takes about 1 human hours and 2 machine hours to preprocess all datasets. To save your time, we've preprocessed all datasets and stored them under `/mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed`.

## ShareGPT

```bash
cd /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/raw

# Download the dataset
wget "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Preprocess the dataset
conda activate loongserve
python3 ~/research/LongServe/test/longserve/4-preprocess-dataset.py --dataset sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --output-path ../sharegpt.ds

# Check the distribution, should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/sharegpt.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ../sharegpt.ds
```

## LEval

```bash
cd /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/raw

# Download the dataset
git clone https://github.com/OpenLMLab/LEval.git

# Preprocess the dataset
conda activate loongserve
python3 ~/research/LongServe/test/longserve/4-preprocess-dataset.py --dataset leval --dataset-path LEval/LEval-data/ --output-path ../leval.ds

# Check the distribution, should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/leval.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ../leval.ds
```

## LV-Eval

```bash
cd /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/raw

# Download the dataset and decompress it
git clone https://huggingface.co/datasets/Infinigence/LVEval
cd LVEval
# We only decompress datasets in English
for file in factrecall_en hotpotwikiqa_mixup loogle_CR_mixup loogle_MIR_mixup loogle_SD_mixup multifieldqa_en_mixup; unzip $file.zip; end
cd ..

# Preprocess the dataset
conda activate loongserve
python3 ~/research/LongServe/test/longserve/4-preprocess-dataset.py --dataset lv-eval --dataset-path LVEval/ --output-path ../lv-eval.ds

# Check the distribution, should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/lv-eval.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ../lv-eval.ds
```

## Mixed

```bash
# Mix the datasets
conda activate loongserve
cd /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/
python3 ~/research/LongServe/test/longserve/4-mix-dataset.py ./mixed1.ds sharegpt.ds:1500 leval.ds:1500 lv-eval.ds:1500

# Check the distribution, should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/mixed1.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ./mixed1.ds
```

## ZipF

```bash
# Mixed the datasets
conda activate loongserve
cd /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/
python3 ~/research/LongServe/test/longserve/4-mix-dataset-zipf.py --output zipf1.0.ds --datasets sharegpt.ds leval.ds lv-eval.ds --num-prompts 20000 --zipf-alpha 1.0
python3 ~/research/LongServe/test/longserve/4-mix-dataset-zipf.py --output zipf1.2.ds --datasets sharegpt.ds leval.ds lv-eval.ds --num-prompts 20000 --zipf-alpha 1.2
python3 ~/research/LongServe/test/longserve/4-mix-dataset-zipf.py --output zipf1.4.ds --datasets sharegpt.ds leval.ds lv-eval.ds --num-prompts 20000 --zipf-alpha 1.4

# Check the distribution
# should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/zipf1.0.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ./zipf1.0.ds
# should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/zipf1.2.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ./zipf1.2.ds
# should be the same as /mnt/petrelfs/zhaoyihao/intlsy/research/datasets/preprocessed/zipf1.4.ds
python3 ~/research/LongServe/test/longserve/4-analyse-dataset.py --dataset-path ./zipf1.4.ds
```
