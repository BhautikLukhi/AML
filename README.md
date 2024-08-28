# pix2pix
Advanced Machine Learning Project


## Overview

This miniproject investigates the Pix2Pix Adversarial learning architecture to transform a given image. There are different variants, ranging from learning to
colorize a monochrome image to replacing a schematic image with a photorealistic one.

## Research Papers Referred
### Image-to-Image Translation with Conditional Adversarial Networks [1]

This is the Research Paper written on the pix2pix network written at the Berkeley AI Research (BAIR) Laboratory, UC Berkeley. This is the research paper that we are trying to replicate and apply to the task of colorizing Black and White images.

### Generative Adversarial Nets [2]

This is the first paper on Generative Adversarial Networks by Ian Goodfellow et al.

## Datasets

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.


## Evaluation Metrics

The evaluation metric that we have used to compare the results is SSIM (Structural Similarity Index) [4] It compares two images and gives an output between 0 and 1. 1 meaning that images are exactly same and 0 meaning that they are completely different. We use it to compare the generated images and the ground truth images.


## How to run

### Creating a dataset

First create a folder `datasets` in the main repository. Create a folder inside that with the name of dataset you are using. Inside that folder, create a folder `full_dataset` that contains all the colored images. After this run the following command.
```bash
python3 create_dataset.py <Dataset Name> <Task>
```
The task can be one of `bnw2color, impaint_fix, deblur`.

### Training the model

Before training, create a folder named `saved_models` in which the results will be automatically saved. After successfully creating the dataset and creating the folder, run the following command to start training the model.
```bash
python3 train.py <Dataset Name> <Task>
```
The hyperparameters can be tuned by making the relevant changes in `train.py`, `generator.py` and `discriminator.py` files.

### Using the trained Model

To use the trained model on eval or test set, use the following command.
```bash
python3 predict.py <Dataset Name> <Task>
```
To switch between eval and test sets make relevant changes in the path variable in the `predict.py` file.

### Evaluation Metrics

Use the `eval_metrics.py` file to get the evaluation results on the images in output folder.

## References:

<ol>
	<li>pix2pix Research Paper: https://arxiv.org/pdf/1611.07004.pdf</li>
	<li>GAN Research Paper: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf</li>
	<li>MIT CVCL Dataset: http://cvcl.mit.edu/database.htm</li>
	<li>SSIM: https://en.wikipedia.org/wiki/Structural_similarity</li>
	<li>Official pix2pix Repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix</li>
</ol>
