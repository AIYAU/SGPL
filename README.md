
# SGPL

Semantic Guided Prototype Learning for Cross-Domain Few-Shot Hyperspectral Image Classification

## Requirements

To run this project, you will need to add the following environment variables to your .env file

`python = 3.8.8`

`torch == 1.12.1+cu113`

`torchvision == 0.13.1+cu113`

## Datasets

- source domain dataset
  - Chikusei
  - Botswana
  - Houston
  - KSC

- target domain datasets
  - Salinas
  - PaviaU
  - Indian Pines
```
An example datasets folder has the following structure:

data
├── SA
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
├── UP
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
└── source_data
    └── the upcoming source domain dataset

The four datasets of the source domain should be stored at locations of your choosing based on the addresses specified in the particular code.

```
## Usage

1. Download the required source and target domain datasets and store them in their specific locations.

2. Preparation of pre_training dataset.
- `generate_source_CH_data_process.py`, `generate_source_HS_data_process.py`,

- `generate_source_HS_data_process.py`, `generate_source_KSC_data_process.py`,

- `generate_source_process.py`. Then you will obtain the datasets required for pre-training.

3. Pre_training.
- `train_SGPL_source.py`.
Then you will obtain the weight of pre-training model.

4. Fine-tuning.
- `train_SGPL_SA.py`,`train_SGPL_UP.py`,`train_SGPL_IP.py`.
Then you will obtain the weight of fine-tuning model.

5. Testing.
- `test_SGPL_SA.py`,`test_SGPL_UP.py`,`test_SGPL_IP.py`.
Then you will obtain the result of different datasets.

## Supplement
To facilitate a faster code execution, we have provided the weights of the pre-trained model in the file. You can directly proceed to the fine-tuning phase, significantly reducing your time.
