# When differential privacy meets NLP: The devil is in the detail

Companion code to our EMNLP 2021 paper.

Pre-print PDF available at arXiv: https://arxiv.org/abs/2109.03175

Please use the following citation

```plain
@InProceedings{Habernal.2021.EMNLP,
    title = {{When differential privacy meets NLP:
              The devil is in the detail}},
    author = {Habernal, Ivan},
    publisher = {Association for Computational Linguistics},
    booktitle = {Proceedings of the 2021 Conference on Empirical
                 Methods in Natural Language Processing},
    pages = {(to appear)},
    year = {2021},
    address = {Punta Cana, Dominican Republic}
}
```

> **Abstract** Differential privacy provides a formal approach to privacy of individuals. Applications of differential privacy in various scenarios, such as protecting users' original utterances, must satisfy certain mathematical properties. Our contribution is a formal analysis of ADePT, a differentially private auto-encoder for text rewriting (Krishna et al, 2021). ADePT achieves promising results on downstream tasks while providing tight privacy guarantees. Our proof reveals that ADePT is not differentially private, thus rendering the experimental results unsubstantiated. We also quantify the impact of the error in its private mechanism, showing that the true sensitivity is higher by at least factor 6 in an optimistic case of a very small encoder's dimension and that the amount of utterances that are not privatized could easily reach 100% of the entire dataset. Our intention is neither to criticize the authors, nor the peer-reviewing process, but rather point out that if differential privacy applications in NLP rely on formal guarantees, these should be outlined in full and put under detailed scrutiny. 

**Contact person**: Ivan Habernal, ivan.habernal@tu-darmstadt.de. https://www.trusthlt.org

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the publication.*


## Setup

Tested with Python 3.8 and `virtualenv`

```bash
$ virtualenv venv --python=python3.8
$ source venv/bin/activate
```

Install packages

```bash
$ pip install -r requirements.txt
```
