# Lmul

Implementation of the $\mathcal{L}$-mul kernel from the paper: [Addition is All You Need](https://arxiv.org/pdf/2410.00907v2)

Benchmarking on Nvidia rtx3090 driver 535.183, CUDA version 12.2, Ubuntu 12.2.

Check out the tutorial for an in depth overview of the method introduced in the paper

**NOTE:** Nvidia GPUs are optimized heavily for FP operations both for quantity and bandwidth. Implementing the following into a CUDA kernel does not give any advantage in terms of computing speed. Pure hardware implementation is the only way to take advantage of the proposed idea.
