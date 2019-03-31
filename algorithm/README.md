# Project Codename 'outfit1' - Alogirthm
A reoccurance of [Interpretable Partitioned Embedding for Customized Fashion Outfit Composition](https://arxiv.org/abs/1806.04845)

## Structure
```json
{
    "network.py": "Implementation of partitioned embedding network",
    "graph.py": "Implementation of composition graph",
    "train.py": "Implementation of model training",
    "app.py": "Kick-start of the whole trained model"
}
```

## Note
Because we use the DeepFashion dataset, the embedding network is now focused on three main attributes: style, shape and remaining. This is slightly different from the original model.
