# Hand Sign Recognition - Alphabets 
## 3-layer shallow network trained on a TF Session
------
### Content
* model.py - preprocessing data, creating placeholders, init params, forward prop, computing loss, and updating params
* utils.py - helper functions for model training
* predict.py - image preprocessing and prediction

### Usage
Train
```
$ python3 model.py num_epochs learning_rate minibatch_size
```
Classify
```
$ python3 predict.py '/path-to-img/img'
```