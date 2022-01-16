# MetaDL self-service : Few-shot learning 
---
This repository contains the code associated to the self-service MetaDL project, based
on the MetaDL competition framework. One can submit a dataset in a tfrecords format 
and obtains the performance of the AAAI 2021 MetaDL competition's winning solution: MetaDelta.

[CodaLab competition link](https://competitions.codalab.org/competitions/31280)

## Outline 
[I - Overview](#i---overview)

[II - Installation](#ii---installation)

[III - References](#iii---references)

---

## I - Overview
This is the official repository of the [Meta-Learning workshop co-hosted competition for AAAI 2021](https://aaai.org/Conferences/AAAI-21/ws21workshops/#ws18). 

## MetaDL competition summary
The competition focus on few-shot learning for image classification. This is an **online competition**, i.e. you need to provide your submission as raw Python code that will be ran on the CodaLab platform. The code is designed to be a module and to be flexible and allows participants to any type of meta-learning algorithms.

You can find more informations on the [ChaLearn website](https://metalearning.chalearn.org/).



## III - References

* [1] - [E. Triantafillou **Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples** -- 2019](https://arxiv.org/pdf/1903.03096)
* [2] - [J. Snell et al. **Prototypical Networks for Few-shot Learning** -- 2017](https://arxiv.org/pdf/1703.05175)
* [3] - [C. Finn et al. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks** -- 2017](https://arxiv.org/pdf/1703.03400)
* [4] - [Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) Science, 350(6266), 1332-1338.
### Disclamer
This module reuses some parts of the recent publication code [E. Triantafillou et al. **Meta-Dataset: GitHub repository**](https://github.com/google-research/meta-dataset) regarging the <u>data generation pipeline</u>. Also the methods in the <code>starting_kit/tutorial.ipynb</code> such as <code>plot_episode()</code>, <code>plot_batch()</code>, <code>iterate_dataset()</code> have been taken from their introduction notebook.