![FairLens Logo](docs/_static/FairLens_759x196.png)

[![CI](https://github.com/synthesized-io/fairlens/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/synthesized-io/fairlens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fairlens)](https://pypi.org/project/fairlens/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/fairlens)](https://pypi.org/project/fairlens)
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://pypi.org/project/fairlens/)
[![Documentation Status](https://readthedocs.org/projects/fairlens/badge/?version=latest)](https://fairlens.readthedocs.io/en/latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub Repo stars](https://img.shields.io/github/stars/synthesized-io/fairlens?style=social)

# FairLens

FairLens is an open source Python library for automatically discovering bias and measuring fairness in data. The package can be used to quickly identify bias, and provides multiple metrics to measure fairness across a range of sensitive and legally protected characteristics such as age, race and sex.


## Bias in my data?
It's very simple to quickly start understanding any biases that may be present in your data.

<img width="50%" align="right" src="https://user-images.githubusercontent.com/13236749/128219642-baeb8577-11cc-4e5a-8a40-0065eb14037a.png">


```python
import pandas as pd
import fairlens as fl

# Load in the data
df = pd.read_csv("datasets/compas.csv")

# Automatically generate a report
fscorer = fl.FairnessScorer(
    df,
    target_attribute="RawScore",
    sensitive_attributes=[
        "Sex",
        "Ethnicity",
        "MaritalStatus"
    ]
)
fscorer.demographic_report()
```
```
Sensitive Attributes: ['Ethnicity', 'MaritalStatus', 'Sex']

                         Group Distance  Proportion  Counts   P-Value
African-American, Single, Male    0.249    0.291011    5902 3.62e-251
      African-American, Single    0.202    0.369163    7487 1.30e-196
                       Married    0.301    0.134313    2724 7.37e-193
        African-American, Male    0.201    0.353138    7162 4.03e-188
                 Married, Male    0.281    0.108229    2195 9.69e-139
              African-American    0.156    0.444899    9023 3.25e-133
                      Divorced    0.321    0.063754    1293 7.51e-112
            Caucasian, Married    0.351    0.049504    1004 7.73e-106
                  Single, Male    0.121    0.582910   11822  3.30e-95
           Caucasian, Divorced    0.341    0.037473     760  1.28e-76

Weighted Mean Statistical Distance: 0.14081832462333957
```

Check out the [documentation](fairlens.readthedocs.io/en/latest) to get started!

See some of our previous blog posts for our take on bias and fairness in ML:

- [Legal consensus regarding biases and fairness in machine learning in Europe and the US](https://www.synthesized.io/post/discrimination-by-artificial-intelligence-2)
- [Fairness and biases in machine learning and their impact on banking and insurance](https://www.synthesized.io/post/fairness-and-biases-in-machine-learning-and-their-impact-on-banking-and-insurance)
- [Fairness and algorithmic biases in machine learning and recommendations to enterprise](https://www.synthesized.io/post/fairness-and-algorithmic-biases-in-machine-learning-and-recommendations)

## Core Features

Some of the main features of Fairlens are:

- **Measuring Bias** - FairLens can be used to measure the extent and significance of biases in datasets using a wide range of statistical distances and metrics.

- **Sensitive Attribute and Proxy Detection** -  Data Scientists may be unaware of protected or sensitive attributes in their data, and potentially hidden correlations between these columns and other non-protected columns in their data. FairLens can quickly identify sensitive columns and flag hidden correlations and the non-sensitive proxies.

- **Visualization Tools** - FairLens has a range of tools that be used to generate meaningful and descriptive diagrams of different distributions in the dataset before delving further in to quantify them. For instance, FairLens can be used to visualize the distribution of a target with respect to different sensitive demographics, or a correlation heatmap.

- **Fairness Scorer** - The fairness scorer is a simple tool which data scientists can use to get started with FairLens. It is designed to just take in a dataset and a target variable and to automatically generate a report highlighting hidden biases, correlations, and containing various diagrams.


The goal of FairLens is to enable data scientists to gain a deeper understanding of their data, and helps to to ensure fair and ethical use of data in analysis and machine learning tasks. The insights gained from FairLens can be harnessed by the [Bias Mitigation](https://www.synthesized.io/post/synthesized-mitigates-bias-in-data) feature of the [Synthesized](https://synthesized.io) platform, which is able to automagically remove bias using the power of synthetic data.


## Installation

FairLens can be installed using pip
```bash
pip install fairlens
```

## Contributing

FairLens is under active development, and we appreciate community contributions. See [CONTRIBUTING.md](https://github.com/synthesized-io/fairlens/blob/main/.github/CONTRIBUTING.md) for how to get started.

The repository's current roadmap is maintained as a Github project [here](https://github.com/synthesized-io/fairlens/projects/1).


## License

This project is licensed under the terms of the [BSD 3](https://github.com/synthesized-io/fairlens/blob/main/LICENSE.md) license.
