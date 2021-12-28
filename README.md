# Heterogeneous interpolation on graph
The ogbg-molhiv and ogbg-molpcba datasets are two molecular property prediction datasets of different sizes: ogbg-molhiv (small) and ogbg-molpcba (medium).
The task is to predict the target molecular properties as accurately as possible, where the molecular properties are cast as binary labels, e.g, whether a molecule inhibits HIV virus replication or not. Note that some datasets (e.g., ogbg-molpcba) can have multiple tasks, and can contain nan that indicates the corresponding label is not assigned to the molecule.
The challenge leaderboard can be checked at: https://ogb.stanford.edu/docs/leader_graphprop/.
We apply Heterogeneous interpolation to solve this challenge and this repo contains our code submission.
The techniqual report can be checked at ./report/.

## Requirements
  Install base packages:
    ```bash
    Python>=3.7
    Pytorch>=1.9.0
    tensorflow>=2.0.0
    pytorch_geometric>=1.6.0
    ogb>=1.3.2
    dgl>=0.5.3
    numpy==1.20.3
    pandas==1.2.5
    scikit-learn==0.24.2
    deep_gcns_torch
    LibAUC
    ```

## Results on OGB Challenges
Running the default code 10 times, here we present our results on the ogbg-molhiv and ogbg-molpcba dataset. For ogbg-molhiv, as the dataset is relatively small, we use DeepGCN as our backbone.

| Dataset | Method             |Test AUROC    |Validation AUROC  | Parameters    | Hardware |
| ------------------ | ------------------ |------------------- | ----------------- | -------------- |----------|
| ogbg-molhiv | DeepGCN+HIG   | 0.8403±0.0021 | 0.8176±0.0034 | 1019408   | Tesla V100 (32GB) |

For ogbg-molpcba, we use Graphormer as our backbone.

| Dataset | Method             |Test AP    |Validation AP  | Parameters    | Hardware |
|------------------- | ------------------ |------------------- | ----------------- | -------------- |----------|
| ogbg-molpcba | Graphormer+HIG   | 0.3167±0.0034 | 0.3252±0.0043 | 119529665   | Tesla V100 (32GB) |



## Training Process for ogbg-molhiv
The training process has three steps: 
1) Preparation: Extract fingerprints and train Random Forest by following [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv)
```
python extract_fingerprint.py
python random_forest.py
```
2) Pretrain: Jointly Train a DeepGCN model with FingerPrints Model.
``` 
python main.py --use_gpu --conv_encode_edge --num_layers 14 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2 \
            --dataset ogbg-molhiv \
            --loss auroc \
            --optimizer pesg \
            --batch_size 512 \
            --lr 0.1 \
            --gamma 500 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 300

```
```
python finetune.py --use_gpu --conv_encode_edge --num_layers 14 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2 \
            --dataset ogbg-molhiv \
            --loss auroc \
            --optimizer pesg \
            --batch_size 512 \
            --lr 0.01 \
            --gamma 300 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 100
```
3) Finetune: Finetune the pretrain model got in Step 2 using NodeDrop Augmentation. We offer a pretrain model in ./saved_models/FT.
```
python finetune_dropnode.py --use_gpu --conv_encode_edge --num_layers 14 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0 \
            --dataset ogbg-molhiv \
            --loss auroc \
            --optimizer pesg \
            --batch_size 512 \
            --lr 0.01 \
            --gamma 300 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 100
```
We offer one sample log file: log_ft_molhiv_20211202.txt.


##Training Process for ogbg-molpcba

The training process has three steps:
```
cd ./Graphormer_with_HIG
```
1) Pretrain: Pretrain a Graphormer model on dataset PCQM4M.
```
sh ./examples/ogb-lsc/lsc-pcba.sh
```
2) Finetune: Finetune the pretrain model got in Step 1 using NodeDrop Augmentation and KL divergence constraint. We offer a pretrain model in ./exps/tmp.

```
sh ./examples/ogb/finetune_dropnode.sh

sh ./examples/ogb/finetune_kl.sh
```


Reference 
---------
- https://libauc.org/
- https://github.com/Optimization-AI/LibAUC
- https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv
- https://github.com/lightaime/deep_gcns_torch/
- https://github.com/yzhuoning/DeepAUC_OGB_Challenge
- https://github.com/microsoft/Graphormer
- https://ogb.stanford.edu/
