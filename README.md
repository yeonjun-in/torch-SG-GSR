# SG-GSR: Self-Guided Robust Graph Structure Refinement

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www2024.thewebconf.org/" alt="Conference">
        <img src="https://img.shields.io/badge/WWW'24-brightgreen" /></a>
    <img src="https://img.shields.io/pypi/l/torch-rechub">
</p>

The official source code for [**Self-Guided Robust Graph Structure Refinement**]() at WWW 2024. 

Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, and Chanyoung Park

## Abstract
Recent studies have revealed that GNNs are vulnerable to adversarial attacks. To defend against such attacks, robust graph structure refinement (GSR) methods aim at minimizing the effect of adversarial edges based on node features, graph structure, or external information. However, we have discovered that existing GSR methods are limited by narrow assumptions, such as assuming clean node features, moderate structural attacks, and the availability of external clean graphs, resulting in the restricted applicability in real-world scenarios. In this paper, we propose a self-guided GSR framework (SG-GSR), which utilizes a clean sub-graph found within the given attacked graph itself. Furthermore, we propose a novel graph augmentation and a group-training strategy to handle the two technical challenges in the clean sub-graph extraction: 1) loss of structural information, and 2) imbalanced node degree distribution. Extensive experiments demonstrate the effectiveness of SG-GSR under various scenarios including non-targeted attacks, targeted attacks, feature attacks, e-commerce fraud, and noisy node labels. 


## Overall Archicteture

<img src="architecture.svg" width="700px"></img> 



### Requirements

- Python version: 3.7.11
- Pytorch version: 1.10.2
- torch-geometric version: 2.0.3
- deeprobust version: 0.2.4

### How to run
You can run the model with following options
- Pretrain structural features (node2vec)
```
sh pretrain_node2vec.sh
```

- To reproduce Table 1 in paper
```
sh train_SGGSR.sh
```

### Data generation
You can generate graph dataset with e-commerce fraud, i.e., Garden and Pet.
```
sh create_ecommerce_fraud.sh
```

Note that before executing the above code, you should download the raw data (i.e., Amazon review data) from the attached [link](http://jmcauley.ucsd.edu/data/amazon/links.html)


### Cite (Bibtex)
- If you find ``SG-GSR`` useful in your research, please cite the following paper:
  - Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, and Chanyoung Park. "Self-Guided Robust Graph Structure Refinement." WWW 2024.
  - Bibtex
```
@article{in2024self,
  title={Self-Guided Robust Graph Structure Refinement},
  author={In, Yeonjun and Yoon, Kanghoon and Kim, Kibum and Shin, Kijung and Park, Chanyoung},
  journal={arXiv preprint arXiv:2402.11837},
  year={2024}
}
```
