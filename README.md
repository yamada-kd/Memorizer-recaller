# Memorizer-Recaller

This repository contains the source codes for generating the memorizer-recaller network, which is proposed in the article, "[Developing appendable memory system towards permanent memory for artificial intelligence to acquire new knowledge after deployment](https://www.biorxiv.org/content/10.1101/2023.05.25.542376v1)."

These codes have been tested and are known to work with Python 3.7.7 and TensorFlow 2.2.0.

## base.01.learning.py
This source code is used subsection 2.2.2, where the memorizer-recaller network is first introduced. The trained parameters can be found in `experiment/base/01`.

## base.02.learning.py
This source code is used in subsection 2.2.1, where we identify problems with the current learning method. The trained parameters can be found in `experiment/base/02`.

## sort.01.learning.py
This source code is used in section 2.3, where we develop a sorting algorithm using the memorizer-recaller network. The trained parameters can be found in `experiment/sort/01`.
