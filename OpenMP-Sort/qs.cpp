#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <chrono>


int Partition(std::vector<int>& A, int l, int r) {
    int p = r;
    int firsthigh = l;
    for (int i = l; i < r; ++i) {
        if (A[i] < A[p]) {
            std::swap(A[i], A[firsthigh]);
            ++firsthigh;
        }
    }
    std::swap(A[firsthigh], A[p]);
    return firsthigh;
}

void QuickSort(std::vector<int>& A, int l, int r) {
    if (A.size() == 0)
        return;
    if (l < r) {
        int pivot = Partition(A, l, r);
#pragma omp task shared(A)
        {
            QuickSort(A, l, pivot - 1);
        }
#pragma omp task shared(A)
        {
            QuickSort(A, pivot + 1, r);
        }
#pragma omp taskwait
    }
}

int main() {
    int N = 10000, a = 0;
    std::vector<int> A, B;
    A.reserve(N);

    while (std::cin >> a) {
        A.push_back(a);
        B.push_back(a);
    }

    std::vector<int> sizes = {1, 2, 4, 8, 16};

    for (auto np : sizes) {
        double res = 0;

        for (int i = 0; i < 10; ++i) {
            A = B;
            auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(np)
#pragma omp single
            QuickSort(A, 0, (int)A.size() - 1);

            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = stop - start;

            res += duration.count();
        }

        res /= 10;

        // for (auto el : A)
        // std::cout << el << " ";

        std::cout << "\n Average time for " << np << " is: " << res  << "\n";
    }
    return 0;
}
