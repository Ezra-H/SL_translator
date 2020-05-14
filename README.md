# Sign language recognition

<!-- ![banner]()

![badge]()
![badge]() -->
[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Translating a sign language video into a word or a sentence.

## Table of Contents

- [Background](#background)
- [DataSetUsed](#DataSetUsed)
- [Install](#install)
- [Usage](#usage)
- [Result](#Result)
- [Contributing](#contributing)

## Background

Sign language is the most basic communication in the daily life of the deaf population.
However, due to the lack of systematic research on sign language, 
it is difficult for many normal people to communicate with the deaf. For the convenience of the deaf population, it is necessary to develope a sign language recognition algorithm. 
Thanks to the development of deep learning, methods based on deep learning made progress in the research of sign language translation algorithm.
The problem of sign language recognition can be transformed into a deep learning method based on Seq2Seq. 
In this project, we address the sign language recognition on a dataset with 500 words. The project is supported by Open Fund of Sun Yat-sen University

## DataSetUsed

The dataset we used was published by USTC. The dataset includes 500 words, 50 videos for each word. For one word, the 50 videos was shoted by different people. This The whole dataset can be downloaded in this [link](http://home.ustc.edu.cn/~pjh/openresources/slr/index.html)

## Install

Most of the modules can be installed by executing the code below.

```
pip install -r requirements.txt
```

Other modules are extra options to install.

### openpose
Click this [link](https://github.com/baidu-research/warp-ctc) to install the openpose.

### warp-ctc

Click this [link](https://github.com/baidu-research/warp-ctc) to install the warp-ctc.

## Usage

### training

Executing the train.sh to train the network

```
sh train.sh
```
You can aslo directly execute the train.py to customize your training.
```
train.py [-h] [--train_src INPUT_DATA_PATH]    
                   [--vail_src EVALUATE_DATA_PATH] [--vocab VOCABULATRY_PATH]    
                   [--feature THE_TYPE_OF_FEATURE_EXTRACTOR]    
                   [--fp16 ACTIVATE_THE_FP16]    
                   ...
```

### evaluating

Executing the vail.sh to evaluate the network

```
sh vail.sh
```

## Results

### 2D Conv results
|  Input Information     | Method+RNN+lr+infeature             |Accuracy|
|------------------------|-------------------------------------|------- |
|  hand+head             | Process_Resnet18_Gru_0.0001_3_2048  | 0.615  | 
|  hand+head             | Process_Resnet34_Gru_0.0001_3_2048  | 0.596  | 
|  hand+head             | Process_Resnet50_Gru_0.0001_3_2048  | 0.65   | 
|  hand+head             | Process_Resnet101_Gru_0.0001_3_2048 | 0.662  | 
|  hand+head             | Process_Resnet152_Gru_0.0001_3_2048 | 0.665  | 
| hand                   | Hand_Resnet18_Gru_0.0001_3_512      | 0.446  | 
| hand                   | Hand_Resnet34_Gru_0.0001_3_512      | 0.432  | 
| hand                   | Hand_Resnet50_Gru_0.0001_3_512      | 0.456  |  
| hand                   | Hand_Resnet101_Gru_0.0001_3_512     | 0.472  |  
| hand                   | Hand_Resnet152_Gru_0.0001_3_512     | 0.436  | 
| Full                   | Orignial_Resnet152_Gru_0.0001_3_512 | 0.356  | 
| Full                   | Orignial_Resnet152_Gru_0.0001_3_512 | 0.357  | 
| Full                   | Orignial_Resnet152_Gru_0.0001_3_512 | 0.364  | 
| Full                   | Orignial_Resnet152_Gru_0.0001_3_512 | 0.368  | 
| Full                   | Orignial_Resnet152_Gru_0.0001_3_512 | 0.371  |  

### 3D Conv results

|  Input Information     | Method          |Accuracy|
|------------------------|-----------------|------- |
|  Full                  | R2Plus1D18_LSTM | 0.666  | 
|  Full                  | P3D19_LSTM      | 0.370  | 

## Contributing
Created by Huang Jianjie, Huang Runhui, Cui liangyu, Chen Zibo, Chen Xuechen.

We welcome improvements from the community, please feel free to submit pull requests.
