# FairLens
[![CI](https://github.com/synthesized-io/fairlens/workflows/CI/badge.svg)](https://github.com/synthesized-io/fairlens/actions)
[![Documentation Status](https://readthedocs.org/projects/fairlens/badge/?version=latest)]()
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


FairLens is an open source Python library for automatically discovering bias and measuring fairlens in data. The package can be used to quickly identify bias, and provides multiple metrics to measure fairlens across a range of sensititive and legally protected characteristics such as age, race and sex.

The goal of FairLens is to enable data scientists to gain a deeper understanding of their data, and helps to to ensure fair and ethical use of data in analysis and machine learning tasks. The insights gained from FairLens can be harnessed by the [Bias Mitigation](https://www.synthesized.io/post/synthesized-mitigates-bias-in-data) feature of the [Synthesized](https://synthesized.io) platform, which is able to automagically remove bias using the power of synthetic data.

See some of our previous blog posts for our take on bias and fairness in ML:

- [Legal consensus regarding biases and fairness in machine learning in Europe and the US](https://www.synthesized.io/post/discrimination-by-artificial-intelligence-2)
- [Fairness and biases in machine learning and their impact on banking and insurance](https://www.synthesized.io/post/fairness-and-biases-in-machine-learning-and-their-impact-on-banking-and-insurance)
- [Fairness and algorithmic biases in machine learning and recommendations to enterprise](https://www.synthesized.io/post/fairness-and-algorithmic-biases-in-machine-learning-and-recommendations)



## Getting Started

fairlens can be installed using pip
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
