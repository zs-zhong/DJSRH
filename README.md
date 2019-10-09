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
- [Ablation studies](#AblationStudies)
- [Comparisons with SOTAs](#ComparisonsWithSOTAs)
***********************************************************************************************************
### Introduction

Cross-modal hashing encodes the multimedia data into a common binary hash space in which the correlations among the samples from different modalities can be effectively measured. Deep cross-modal hashing further improves the retrieval performance as the deep neural networks can generate more semantic relevant features and hash codes. In this paper, we study the unsupervised deep cross-modal hash coding and propose Deep JointSemantics Reconstructing Hashing (DJSRH), which has the following two main advantages. First, to learn binary codes that preserve the neighborhood structure of the original data, DJSRH constructs a novel joint-semantics affinity matrix which elaborately integrates the original neighborhood information from different modalities and accordingly is capable to capture the latent intrinsic semantic affinity for the input multi-modal instances. Second, DJSRH later trains the networks to generate binary codes that maximally reconstruct above joint-semantics relations via the proposed reconstructing framework, which is more competent for the batch-wise training as it reconstructs the specific similarity value unlike the common Laplacian constraint merely preserving the similarity order. Extensive experiments demonstrate the significant improvement by DJSRH in various cross-modal retrieval tasks.

<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/DJRSH.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************
### Usage


***********************************************************************************************************
### Ablation studies


***********************************************************************************************************
### Comparisons with SOTAs


***********************************************************************************************************
### Citation

If you find this page useful, please cite our paper:

	@inproceedings{su2019deep,
		title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval},
		author={Shupeng su, Zhisheng Zhong, Chao Zhang},
		booktitle={International Conference on Computer Vision},
		year={2019}
	}

All rights are reserved by the authors.
***********************************************************************************************************