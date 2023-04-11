# PromptDA: Label-guided Data Augmentation for Prompt-based Few-Shot Learners

The code for **[PromptDA: Label-guided Data Augmentation for Prompt-based Few-Shot Learners](https://arxiv.org/abs/2205.09229)**

[**Canyu Chen**](https://canyuchen.com), [Kai Shu](http://www.cs.iit.edu/~kshu/)

Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (**EACL 2023** Main Conference Long Paper). 

[[Paper]](https://arxiv.org/abs/2205.09229)
[[Slides]](https://drive.google.com/file/d/1aqUdZ6B0-tqjDp_HE5VH5BPkY0B7mucc/view?usp=share_link)
[[Poster]](https://drive.google.com/file/d/1t-n811sq1897CvHHe3w_QAgUtyGZaN68/view?usp=share_link)


## Basic Idea
<img src="./image/image2.jpg" width="500">    

The basic comparison of conventional data augmentation methods **Synonym Substitution** and our proposed augmentation framework **PromptDA**. `good, great, fine` are the label words for prompt tuning. Conventional DA constructs *`instances`* for augmentation, but PromptDA constructs *`instance-label pairs`* for augmentation.


## Framework
<img src="./image/image1.jpg" width="800">    

The proposed **PromptDA** for few-shot learning (with sentiment classification task as an example): (a) **Label Augmentation**: deriving multiple label words for each class to enrich the label semantic space; (b) **Augmented Prompt-based Tuning**: training with the augmented instance-label pairs via masked language modeling; (c) **Prediction Transformation**:  aggregating  the probability scores on the derived label words for the final prediction. 

## Quick Start

First, download our repo:
```bash
git clone https://github.com/canyuchen/PromptDA.git
cd PromptDA
```
Evironment setup (Note that we run the experiments on one NVIDIA GeForce RTX 3090 GPU. We trained the model with the torch version = 1.10.1+cu113 and CUDA Version = 11.4. Results could be different with other torch and CUDA versions.):
```bash
conda create -n promptda python=3.6.13
pip install -r requirements.txt
```
Reproduce the results for standard prompt tuning with template:
```bash
cd test_standard_with_template
CUDA_VISIBLE_DEVICES=0 sh test.sh
```
Reproduce the results for prompt tuning with template + PromptDA with manual label augmentation:
```bash
cd test_promptda_template_manual
CUDA_VISIBLE_DEVICES=0 sh test.sh
```
Reproduce the results for prompt tuning with template + PromptDA with automatic label augmentation:
```bash
cd test_promptda_template
CUDA_VISIBLE_DEVICES=0 sh test.sh
```
It is worth noting that the results of my training process on `CoLA`, `CR`, `MR`, `MPQA`, `SST-2`, `SST-5`, `SUBJ` datasets are recorded in the `*.log` files in `test_standard_with_template/test_*`, `test_promptda_template_manual/test_*`, and `test_promptda_template/test_*`.


## Label Augmentation

The evironment setup for Label Augmentation is the same as [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

Reproduce automatic label augmentation:
```bash
cd LM-BFF/tools
sh run_generate_labels.sh
```
It is worth noting that the results of our automatic label augmentation on `CoLA`, `CR`, `MR`, `MPQA`, `SST-2`, `SST-5`, `SUBJ` datasets are recorded in the files in `LM-BFF/my_auto_label_mapping/manual_template`.

## Results
<img src="./image/result.jpg" width="800">  


## Citation
If you find our code useful, we will greatly appreacite it if you could consider citing our paper:
```
@article{chen2022promptda,
  title   = {PromptDA: Label-guided Data Augmentation for Prompt-based Few Shot Learners},
  author  = {Canyu Chen and Kai Shu},
  year    = {2022},
  journal = {arXiv preprint arXiv: Arxiv-2205.09229}
}
```

## Acknowledgement

Our Label Augmentation method is modified based on the code of Making Pre-trained Language Models Better Few-shot Learners ([LM-BFF](https://github.com/princeton-nlp/LM-BFF), ACL 2021). Thanks to the authors!









