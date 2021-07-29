# FairLens
[![CI](https://github.com/synthesized-io/fairlens/workflows/CI/badge.svg)](https://github.com/synthesized-io/fairlens/actions)
[![Documentation Status](https://readthedocs.org/projects/fairlens/badge/?version=latest)]()
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


FairLens is an open source Python library for automatically discovering bias and measuring fairness in data. The package can be used to quickly identify bias, and provides multiple metrics to measure fairness across a range of sensitive and legally protected characteristics such as age, race and sex.


## Installation

FairLens is compatible with python versions 3.6+ and can be installed using pip
```bash
pip install fairlens
```

## Getting Started

```python
import pandas as pd
import fairlens as fl

# Load in the data
df = pd.read_csv("datasets/compas.csv")

fscorer = fl.FairnessScorer(df, "RawScore")

fscorer.report()
```

## Core Features

Some of the main features of Fairlens are:

- **Measuring Bias** - FairLens can be used to measure the extent and significance of biases in datasets using a wide range of statistical distances and metrics.

- **Sensitive Attribute and Proxy Detection** -  Data Scientists may be unaware of protected or sensitive attributes in their data, and potentially hidden correlations between these columns and other non-protected columns in their data. FairLens can quickly identify sensitive columns and flag hidden correlations and the non-sensitive proxies.

- **Visualization Tools** - FairLens has a range of tools that be used to generate meaningful and descriptive diagrams of different distributions in the dataset before delving further in to quantify them. For instance, FairLens can be used to visualize the distribution of a target with respect to different sensitive demographics, or a correlation heatmap.

- **Fairness Scorer** - The fairness scorer is a simple tool which data scientists can use to get started with FairLens. It is designed to just take in a dataset and a target variable and to automatically generate a report highlighting hidden biases, correlations, and containing various diagrams.

- **Documentation** - Fairlens has a comprehensive documentation, containing user guides for the most interesting and useful features, as well as a complete auto-generated API reference, documenting all of the contributions and changes to the package.



The goal of FairLens is to enable data scientists to gain a deeper understanding of their data, and helps to to ensure fair and ethical use of data in analysis and machine learning tasks. The insights gained from FairLens can be harnessed by the [Bias Mitigation](https://www.synthesized.io/post/synthesized-mitigates-bias-in-data) feature of the [Synthesized](https://synthesized.io) platform, which is able to automagically remove bias using the power of synthetic data.

See some of our previous blog posts for our take on bias and fairness in ML:

- [Legal consensus regarding biases and fairness in machine learning in Europe and the US](https://www.synthesized.io/post/discrimination-by-artificial-intelligence-2)
- [Fairness and biases in machine learning and their impact on banking and insurance](https://www.synthesized.io/post/fairness-and-biases-in-machine-learning-and-their-impact-on-banking-and-insurance)
- [Fairness and algorithmic biases in machine learning and recommendations to enterprise](https://www.synthesized.io/post/fairness-and-algorithmic-biases-in-machine-learning-and-recommendations)



## Getting Started

FairLens can be installed using pip
```bash
pip install fairlens
```

### Usage
```python
import fairlens
```

## Contributing

FairLens is under active development, and we appreciate community contributions. See [CONTRIBUTING.md](https://github.com/synthesized-io/fairlens/blob/main/.github/CONTRIBUTING.md) for how to get started.


## License

This project is licensed under the terms of the [BSD 3](https://github.com/synthesized-io/fairlens/blob/main/LICENSE.md) license.
