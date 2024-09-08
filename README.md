# GC_N_prediction

## Overview
This repository contains the implementation of the GC Segmentation algorithm and the construction of a predictive model. 

## GC Segmentation

Our pre-developed model for gastric cancer segmentation, ALIEN(https://github.com/ZHChen-294/ALIEN), was created by Zhihong Chen, Lisha Yao, Yanfen Cui, Yunlin Zheng, Suyun Li, Xiaorui Han, Xuewei Kang, Xin Chen, Wenbin Liu, Chu Han, Zaiyi Liu, Bingjiang Qiu, Gang Fang

Paper: **ALIEN**: **A**ttention-Guided Cross-Reso**L**ut**I**on Collaborativ**E** **N**etwork for 3D Gastric Cancer Segmentation in CT Images
<!-- Submitted to [**Biomedical Signal Processing and Control**](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) (In Revising). -->

* An attention-guided collaborative network for 3D gastric cancer segmentation in CT images.
* Multi-attention fusion module enhances the encoding capacity, addressing the issue of over-segmentation.
* Cross-resolution fusion module facilitates the simultaneous capture of detailed and high-level semantic information.
* Scale-aware activation module selectively extracts and integrates specific patterns from decoded features for feature refinement.

## Model Construction

The `model build.py` file contains model-building and prediction codes, which can be used to output prediction values.


