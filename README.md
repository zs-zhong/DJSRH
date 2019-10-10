# DJSRH
***********************************************************************************************************

This repository is for "Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval" (to appear in ICCV 2019, Oral)

By Shupeng Su\*, [Zhisheng Zhong](https://zzs1994.github.io)\*, Chao Zhang (\* Authors contributed equally).

For more details or questions, feel free to contact: 

Zhisheng Zhong (zszhong@pku.edu.cn) and Shupeng Su (sushupeng@pku.edu.cn)

***********************************************************************************************************
### Table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Ablation studies](#ablation-studies)
- [Comparisons with SOTAs](#comparisons-with-SOTAs)
***********************************************************************************************************

### Introduction

Cross-modal hashing encodes the multimedia data into a common binary hash space in which the correlations among the samples from different modalities can be effectively measured. Deep cross-modal hashing further improves the retrieval performance as the deep neural networks can generate more semantic relevant features and hash codes. In this paper, we study the unsupervised deep cross-modal hash coding and propose Deep JointSemantics Reconstructing Hashing (DJSRH), which has the following two main advantages. First, to learn binary codes that preserve the neighborhood structure of the original data, DJSRH constructs a novel joint-semantics affinity matrix which elaborately integrates the original neighborhood information from different modalities and accordingly is capable to capture the latent intrinsic semantic affinity for the input multi-modal instances. Second, DJSRH later trains the networks to generate binary codes that maximally reconstruct above joint-semantics relations via the proposed reconstructing framework, which is more competent for the batch-wise training as it reconstructs the specific similarity value unlike the common Laplacian constraint merely preserving the similarity order. Extensive experiments demonstrate the significant improvement by DJSRH in various cross-modal retrieval tasks.

<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/DJRSH.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************

### Usage
To DO

***********************************************************************************************************

### Ablation studies
Table 1. The mAP@50 results on NUS-WIDE to evaluate the effectiveness of each component in DJSRH.

Model|Configuration|64bits (I->T)|64bits (T->I)|128bits (I->T)|128bits (T->I)|
|:---------:|:---:|:-----:|:----:|:----:|:----:|
DJSRH-1|S=S_I|0.717|0.712|0.741|0.735|
DJSRH-2|S=S_T|0.702|0.606|0.734|0.581|
DJSRH-3|βS_I+(1−β)S_T|0.724|0.720|0.747|0.738|
DJSRH-4|+(η=0.4)|0.790|0.745|0.803|0.757|
DJSRH-5|+(µ=1.5)|0.793|0.747|0.812|0.768|
DJSRH|+(λ_1=λ_2=0.1)|0.798|0.771|0.817|0.789|
DJSRH-6|−(α=1)|0.786|0.770|0.811|0.782|

From the table we can observe that each of our proposed components plays a certain role for our final results. We would like to highlight that the variants DJSRH-1,2,3 have surpassed UDCMH (the state of-the-art previous method in Table 2).

***********************************************************************************************************

### Comparisons with SOTAs
To DO

***********************************************************************************************************
### Citation

If you find this code useful, please cite our paper:

	@inproceedings{su2019deep,
		title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval},
		author={Shupeng Su, Zhisheng Zhong, Chao Zhang},
		booktitle={International Conference on Computer Vision},
		year={2019}
	}

All rights are reserved by the authors.
***********************************************************************************************************