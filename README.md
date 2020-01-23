# Robust Learning with the Hilbert-Schmidt Independence Criterion

This repository contains a pytorch implementation of HSIC-loss used in the paper https://arxiv.org/abs/1910.00270.

If ```x,y``` represent two batches of samples from random variables ```X,Y```, calling
```
HSIC(x,y,s_x,s_y)
```
would compute the Hilbert-Schmidt Independence Criterion between them.
This code uses Gaussian kernels, and the parameters ``` s_x,s_y ``` represent their width.
