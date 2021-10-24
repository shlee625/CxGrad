# CxGrad - Official PyTorch Implementation
> **Contextual Gradient Scaling for Few-Shot Learning**<br>
> **Sanghyuk Lee**, Seunghyun Lee, and Byung Cheol Song<br>
> In WACV 2022.<br>
Paper: [https://arxiv.org/abs/2110.10353](https://arxiv.org/abs/2110.10353)<br>

<!-- > **Abstract:** *Model-agnostic meta-learning (MAML) is a well-known optimization-based meta-learning algorithm that works well in various computer vision tasks, e.g., few-shot classification. MAML is to learn an initialization so that a model can adapt to a new task in a few steps. However, since the gradient norm of a classifier (head) is much bigger than those of backbone layers, the model focuses on learning the decision boundary of the classifier with similar representation. Furthermore, gradient norms of high-level layers are small than those of the other layers. So, the backbone of MAML usually learns task-generic features, which results in deteriorated adaptation performance in the inner-loop. To resolve or mitigate this problem, we propose contextual gradient scaling (CxGrad), which scales gradient norms of the backbone to facilitate learning task-specific knowledge in the inner-loop. Since the scaling factors are generated from task-conditioned parameters, gradient norms of the backbone can be scaled in a task-wise fashion. Experimental results show that CxGrad effectively encourages the backbone to learn task-specific knowledge in the inner-loop and improves the performance of MAML up to a significant margin in both same- and cross-domain few-shot classification.* -->

## Installation
This code is based on PyTorch. Please make a virtual environment and use it by running the command below:
```bash
conda env create --file environment.yml -n CxGrad
conda activate CxGrad
```

## Datasets
We provide instructions to download 4 datasets: `miniImageNet`, `tieredImageNet`, `CUB`, and `CIFAR-FS`. Download the datasets you want to use and move them to `datasets`.

1. miniImageNet: Download `mini_imagenet_full_size.tar.bz2` from this [link](https://drive.google.com/file/d/1qQCoGoEJKUCQkk8roncWH7rhPN7aMfBr/view), provided in [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch). Note that by downloading and using the miniImageNet, you accept terms and conditions found in `imagenet_license.md`.

2. tieredImageNet: Download `tiered_imagenet.tar` from this [link](https://mtl.yyliu.net/download/Lmzjm9tX.html).

3. CIFAR-FS: Download `cifar100.zip` from this [link](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view?usp=sharing). The splits and the download link are provided by [Bertinetto](https://github.com/bertinetto/r2d2).

4. CUB: Download `CUB_200_2011.tgz` from this [link](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz). The classes of each split are randomly chosen. Thus, we provide the splits of our experiments: `CUB_split_train.txt`, `CUB_split_val.txt`, and `CUB_split_test.txt` in `datasets/preprocess`. These splits are done by a script written by [Chen](https://github.com/wyharveychen/CloserLookFewShot).

Then, run the command below to preprocess the datasets you downloaded.
```bash
python preprocess/preprocess.py --datasets DATASET1 DATASET2 ...
```

The structure should be like this:
   ```
   CxGrad 
     ├── datasets
     |      ├── miniImageNet
     |      |        ├── train
     |      |        ├── val
     |      |        └── test
     |      |── tieredImageNet
     |      |         ├── train
     |      |         ├── val
     |      |         └── test
     |      ├── CIFAR-FS
     |      |       ├── train
     |      |       ├── val
     |      |       └── test
     |      └── CUB
     |           ├── train
     |           ├── val
     |           └── test
     ├── utils
     ├── README.md
     └── ...
   ```

## Run experiments
- Change directory to `experiment_scripts`.

### Train
- In order to train the model on N-way K-shot miniImageNet classification, run
   ```
   bash mini_imagenet_Nway_Kshot/CxGrad_4conv.sh GPU_ID
   ```
- Otherwise for tieredImageNet, run
  ```
   bash tiered_imagenet_Nway_Kshot/CxGrad_4conv.sh GPU_ID
  ```

### Test
- ex) Test on CUB using the model trained on 5-way 5-shot miniImageNet
  ```
   TEST=1 TEST_DATASET=CUB bash mini_imagenet_5way_5shot/CxGrad_4conv.sh GPU_ID
  ```

## Citation
To be prepared

## Acknowledgments
Thanks to the authors of [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) and [ALFA](https://github.com/baiksung/ALFA), which our work is based on, for their great implementations.
