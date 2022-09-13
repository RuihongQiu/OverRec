# OverRec
Code for ICDM 2022 paper, [Beyond Double Ascent via Recurrent Neural Tangent Kernel in Sequential Recommendation](https://arxiv.org/abs/2209.03735).

# Usage

Download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Clothing_Shoes_and_Jewelry
│   ├── Amazon_Clothing_Shoes_and_Jewelry.inter
│   └── Amazon_Clothing_Shoes_and_Jewelry.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
├── ml-1m
│   ├── ml-1m.inter
│   ├── ml-1m.item
│   ├── ml-1m.user
│   └── README.md
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user

```

Run `python3 run_overrec.py`.

This experiment does not require GPU calculation. It is purely CPU computation. Yet it may require large memory for kernel calculation, ranging from 100 Gb ~ 300 Gb for different datasets.

# MISC
We implement [SKNN](https://dl.acm.org/doi/10.1145/3109859.3109872) and [STAN](https://dl.acm.org/doi/10.1145/3331184.3331322) in this repo.
Run `python3 run_sknn.py` or `python3 run_stan.py`.

# Cite

If you find this repo useful, please cite
```
@article{OverRec,
  author    = {Ruihong Qiu and
               Zi Huang and
               Hongzhi Yin},
  title     = {Beyond Double Ascent via Recurrent Neural Tangent Kernel in Sequential Recommendation},
  journal   = {CoRR},
  volume    = {abs/2209.03735},
  year      = {2022},
}
```

# Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [RNTK](https://github.com/SinaAlemohammad/RNTK_UCI)