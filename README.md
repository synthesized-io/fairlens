![FairLens Logo](docs/_static/FairLens_759x196.png)

[![CI](https://github.com/synthesized-io/fairlens/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/synthesized-io/fairlens/actions/workflows/ci.yml)
![PyPI](https://img.shields.io/pypi/v/fairlens)
![PyPI - Downloads](https://img.shields.io/pypi/dw/fairlens)
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)]()
[![Documentation Status](https://readthedocs.org/projects/fairlens/badge/?version=latest)]()
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub Repo stars](https://img.shields.io/github/stars/synthesized-io/fairlens?style=social)

# FairLens

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

# Automatically generate a report
fscorer = FairnessScorer(
    df,
    target_attribute="RawScore",
    sensitive_attributes=["Sex", "Ethnicity", "MaritalStatus", "Language"]
)
fscorer.report()

Sensitive Attributes: Sex, Ethnicity, MaritalStatus, Language

                             Group  Distance  Proportion  Counts
0                             Male  0.162364    0.780928   15838
1                           Female  0.162364    0.219072    4443
2                        Caucasian  0.166039    0.358020    7261
3                 African-American  0.281050    0.444899    9023
4                         Hispanic  0.190989    0.143681    2914
5                           Single  0.362521    0.741679   15042
6                         Married  0.347185    0.134313    2724
7                  Male, Caucasian  0.119241    0.268724    5450
8           Male, African-American  0.310895    0.353138    7162
9                   Male, Hispanic  0.153909    0.115231    2337
10                    Male, Single  0.289278    0.582910   11822
11                   Male, Married  0.314981    0.108229    2195
12                  Female, Single  0.088391    0.158769    3220
13               Caucasian, Single  0.067940    0.249790    5066
14        African-American, Single  0.320650    0.369163    7487
15         Male, Caucasian, Single  0.050034    0.192545    3905
16  Male, African-American, Single  0.351865    0.291011    5902

Overall Fairness Rating: 0.7617094043914152
```

## Core Features

Some of the main features of Fairlens are:

- **Measuring Bias** - FairLens can be used to measure the extent and significance of biases in datasets using a wide range of statistical distances and metrics.

- **Sensitive Attribute and Proxy Detection** -  Data Scientists may be unaware of protected or sensitive attributes in their data, and potentially hidden correlations between these columns and other non-protected columns in their data. FairLens can quickly identify sensitive columns and flag hidden correlations and the non-sensitive proxies.

- **Visualization Tools** - FairLens has a range of tools that be used to generate meaningful and descriptive diagrams of different distributions in the dataset before delving further in to quantify them. For instance, FairLens can be used to visualize the distribution of a target with respect to different sensitive demographics, or a correlation heatmap.

- **Fairness Scorer** - The fairness scorer is a simple tool which data scientists can use to get started with FairLens. It is designed to just take in a dataset and a target variable and to automatically generate a report highlighting hidden biases, correlations, and containing various diagrams.


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
