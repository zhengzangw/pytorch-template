<div align="center">

# Your Project Name

<!--
ARXIV
[![Paper](https://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!--
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--
Conference
[![Conference](https://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](https://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](https://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
-->
</div>

## Description

TBD

## How to run

First, install dependencies

```bash
# clone project
git clone https://github.com/zhengzangw/this_project

# install project
cd this_project
pip install -e .
pip install -r requirements.txt
```

Next, run the training or testing script.

```bash
cd project

# train
python -m project.train --config config/debug.yaml

# test
python -m project.train --config config/test.yaml
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

## Citation

```bib
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
