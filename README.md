
# What's this
Implementation of Swapout by chainer  

# Dependencies

    git clone https://github.com/nutszebra/swapout.git
    cd swapout
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Stochastic inference  
Implemented


# Cifar10 result

| network                        | depth | k  | total accuracy (%) |
|:-------------------------------|-------|----|-------------------:|
| Swapout v2(20) Wx4[[1]][Paper] | 20    | 4  | 94.91              |
| Swapout v2(32) Wx4[[1]][Paper] | 32    | 4  | 95.24              |
| my implementation              | 32    | 4  | 95.34              |

<img src="https://github.com/nutszebra/swapout/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/swapout/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Swapout: Learning an ensemble of deep architectures [[1]][Paper]

[paper]: https://arxiv.org/abs/1605.06465 "Paper"
