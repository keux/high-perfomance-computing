#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <queue>


void Merge(std::vector<int>& A, int l, int r, int mid) {
    std::queue<int> left, right;
    for (int i = l; i <= mid; ++i) {
        left.push(A[i]);
    }
    for (int i = mid + 1; i <= r; ++i) {
        right.push(A[i]);
    }
    int i = l;
    while(!(left.empty() || right.empty())) {
        if (left.front() <= right.front()) {
            A[i++] = left.front();
            left.pop();
        } else {
            A[i++] = right.front();
            right.pop();
        }
    }
    while(!left.empty()) {
        A[i++] = left.front();
        left.pop();
    }
    while(!right.empty()) {
        A[i++] = right.front();
        right.pop();
    }
}

void MergeSort(std::vector<int>& A, int l, int r) {
    if (A.size() == 0)
        return;
    if (l < r) {
        int mid = (l + r) / 2;

#pragma omp task shared(A)
        {
            MergeSort(A, l, mid);
        }
#pragma omp task shared(A)
        {
            MergeSort(A, mid + 1, r);
        }
#pragma omp taskwait
        Merge(A, l, r, mid);
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
            MergeSort(A, 0, (int)A.size() - 1);

            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = stop - start;

            res += duration.count();
        }

        res /= 10;

        // for (auto el : A)
        //  std::cout << el << " ";

        std::cout << "\n Average time for " << np << " is: " << res  << "\n";
    }

    return 0;
}
