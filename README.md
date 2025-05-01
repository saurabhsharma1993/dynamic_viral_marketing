# dynamic_viral_marketing
Code for our paper Dynamic Gradient Influencing for Viral Marketing Using Graph Neural Networks, published at the WebConf25.

# Abstract
<img src="https://github.com/saurabhsharma1993/dynamic_viral_marketing/blob/main/data/figures/Figure 1.png" width="1000">
The problem of maximizing the adoption of a product through viral marketing in social networks has been studied heavily through postulated network models. We present a novel data-driven formulation of the problem. We use Graph Neural Networks (GNNs) to model the adoption of products by utilizing both topological and attribute information. The resulting Dynamic Viral Marketing (DVM) problem seeks to find the minimum budget and minimal set of dynamic topological and attribute changes in order to attain a specified adoption goal. We show that DVM is NP-Hard and is related to the existing influence maximization problem. Motivated by this connection, we develop the idea of Dynamic Gradient Influencing (DGI) that uses gradient ranking to find optimal perturbations and targets low-budget and high influence non-adopters in discrete steps. We use an efficient strategy for computing node budgets and develop the “Meta-Influence” heuristic for assessing a node’s downstream influence. We evaluate DGI against multiple baselines and demonstrate gains on average of 24% on budget and 37% on AUC on real-
world attributed networks. 
<img src="https://github.com/saurabhsharma1993/dynamic_viral_marketing/blob/main/data/figures/Figure 2.png" width="1000"> 

# Running the code
1. Clone the conda environment
```
conda env create -f environment.yml
```

2. Run the Dynamic Gradient Influencing (DGI) pipeline,
```
python spread_efficient.py --dataset [name_of_dataset] --exp [name_of_exp] 
```
The experiment logs and results will be dumped in,
```
./logs/[name_of_dataset]/[name_of_exp]
```
Specify datasets using --dataset [dataset]

Use --small flag to get our splits.

Specify GNN model using --model [model]

Specify number of meta attribute flips using --switch_k [value]

Specify influence threshold using --[influence_thresh] [value]

# Citing 
If you use this code, please cite our work:
```
@inproceedings{sharma2025dynamic,
  title={Dynamic Gradient Influencing for Viral Marketing Using Graph Neural Networks},
  author={Sharma, Saurabh and Singh, Ambuj},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={3982--3993},
  year={2025}
}
```
