<div align=center>
  
# ControlCap: Controllable Region-level Captioning
</div>

This is the official implementaion of paper [***ControlCap: Controllable Region-level Captioning***](https://arxiv.org/pdf/2401.17910.pdf). This repository contains Pytorch training code, evaluation code, pre-trained models, and visualization method.

<div align=center>

[![arXiv preprint](http://img.shields.io/badge/arXiv-2307.09756-b31b1b)](https://arxiv.org/pdf/2401.17910.pdf)
![Python 3.8](https://img.shields.io/badge/Python-3.8-green.svg?style=plastic)
![PyTorch 1.11](https://img.shields.io/badge/PyTorch-1.11-EE4C2C.svg?style=plastic)
[![LICENSE](https://img.shields.io/github/license/vasgaowei/ts-cam.svg)](LICENSE)


The code is being cleaned up and the finished code will be updated in the future.


## 2. Introduction

Region-level captioning is challenged by the caption degeneration issue, which refers to that pre-trained multimodal models tend to predict the most frequent captions but miss the less frequent ones. In this study, we propose a controllable region-level captioning (ControlCap) approach, which introduces control words to a multimodal model to address the caption degeneration issue. In specific, ControlCap leverages a discriminative module to generate control words within the caption space to partition it to multiple sub-spaces. The multimodal model is constrained to generate captions within a few sub-spaces containing the control words, which increases the opportunity of hitting less frequent captions, alleviating the caption degeneration issue. Furthermore, interactive control words can be given by either a human or an expert model, which enables captioning beyond the training caption space, enhancing the model’s generalization ability. Extensive experiments on Visual Genome and RefCOCOg datasets show that ControlCap respectively improves the CIDEr score by 21.6 and 2.2, outperforming the state-of-the-arts by significant margins.


## 5. Contacts
If you have any question about our work or this repository, please don't hesitate to contact us by emails or open an issue under this project.
- [zhaoyuzhong20@mails.ucas.ac.cn](zhaoyuzhong20@mails.ucas.ac.cn)
- [wanfang@ucas.ac.cn](wanfang@ucas.ac.cn)

## 6. Acknowledgment

- Part of the code is borrowed from [LAVIS](https://github.com/salesforce/LAVIS), [GlaMM](https://github.com/mbzuai-oryx/groundingLMM), and [RAM](https://github.com/xinyu1205/recognize-anything), we sincerely thank them for their contributions to the community.
