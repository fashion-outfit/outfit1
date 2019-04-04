# Project Codename 'outfit1' - Alogirthm
A reoccurance of [Interpretable Partitioned Embedding for Customized Fashion Outfit Composition](https://arxiv.org/abs/1806.04845). DeepFashion dataset download can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). Polyvore dataset download can be fount [here](https://github.com/xthan/polyvore-dataset)

## Structure
```json
{
    "network.py": "Implementation of partitioned embedding network",
    "graph.py": "Implementation of composition graph",
    "train.py": "Define steps of model training",
    "app.py": "Kick-start of the whole trained model",
    "preprocess.py": "Data preprocessing"
}
```

## Note
1. Using the DeepFashion dataset, this implementation is now focusing on three main attributes: style, shape and remaining. This is slightly different from the original model.

2. Furtherly dividing auto-encoder module into two parts: the encoder part and decoder part. No longer treating them as one. We make this modification to ease parameter sharing.

3. 'Style feature vector' and 'shape feature vector' need to be constructed manually.

4. Attributes network that mentioned in the original paper cannot be implemented due to the DeepFashion dataset we use. This part will now be implemented using Softmax or multi-class SVM instead.
