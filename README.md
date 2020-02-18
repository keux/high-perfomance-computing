# High-Performance-Computing-Course

#### OpenMp Sorting

1. Insertion sort: finds a place for the input element in the sorted subarray by parallel alg and shifts all elements right.
2. Selection sort: declares a compare function.
3. Quick sort, Merge sort: simple.

#### CUDA Gaussian Blur

1. main.cu : the main programm which reads image from file, calculates gaussian kernel, runs it on CUDA and writes blurred result to another file. All parameters are changeble. Filter size is an odd number!
2. notebook contains results for different tests and also it converts img-rgb, rgb-img.

#### parallel-heat-equation
1. solves 1d heat equation in parallel using MPI
