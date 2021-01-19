# NLP_TopicModeling

### Introduction
This is an implementation of LDA(Latent Dirichlet allocation) topic modeling as one of my project in QUT IFN619.

### Data
Dataset used here is 'million ABC news headlines' sources from https://www.kaggle.com/therohk/million-headlines. A million news headlines and its published date are provided in this dataset from 2003 until 2020.

### Methodology
To achieve our goal, firstly, a tf/tfidf model will be built based on tri-gram phrases with python scikit-learn library, then a topic modelling technique will be applied. Topic modelling can be described as a method for finding a group of words (i.e topic) from a collection of documents that best represents the information in the collection. In this project, the LDA (Latent Dirichlet allocation ) will be chosen as the techniques for our topic. The evaluation and visualization will be also given.
