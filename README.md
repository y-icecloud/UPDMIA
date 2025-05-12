# UPDMIA
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
This repo is for source code of "UPDMIA: Unified Processing of Diverse Medical Imaging Data via a Multi-Channel EfficientNet Architecture".
# A Gentle Introduction

<div align="center">
  <img src="https://github.com/y-icecloud/UPDMIA/UPDMIA_Overview.png" alt="Framework">
</div>

Pulmonary medical image analysis—spanning CT, MRI, PET, X-ray, and histopathological slides—has become increasingly crucial for early diagnosis and disease management, yet traditional methods relying on manual segmentation, thresholding, and handcrafted features struggle with inefficiency, low robustness, and poor generalization across imaging modalities. Motivated by the success of deep learning in capturing complex patterns and automating feature extraction, can we develop a unified framework that overcomes these limitations and enables accurate, scalable, and modality-adaptive pulmonary diagnostics? To this end, we propose a flexible deep learning pipeline that integrates a structurally enhanced EfficientNet-based backbone for high-fidelity multi-modal representation learning, a modality-adaptive head that dynamically adjusts to CT, MRI, PET, X-ray, or histopathology inputs, and a dynamic augmentation scheme combined with self-stabilizing optimization to improve robustness. Additionally, we construct the most comprehensive multi-modal pulmonary dataset to date, comprising over 40,000 labeled samples across five imaging types, enabling robust cross-modal training and evaluation. 

