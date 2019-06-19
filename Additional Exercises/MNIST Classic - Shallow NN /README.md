# MNIST Digits Benchmark  
## 3-layer shallow network trained on a TF Session
------
### Content
* model.py - preprocessing data, creating placeholders, init params, forward prop, computing loss, and updating params
* utils.py - helper functions for model training
* predict.py - image preprocessing and prediction

### Performance
Over 2000 epochs on mini-batches with ADAM Optimizer:
```
Train Accuracy: 0.9974667
Test Accuracy: 0.9597
```
### Usage
Train
```
$ python3 model.py num_epochs learning_rate minibatch_size
```
Classify
```
$ python3 predict.py '/path-to-img/img'
```
