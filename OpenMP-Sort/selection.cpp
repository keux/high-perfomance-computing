#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <chrono>

struct Compare { int val; int index; };
#pragma omp declare reduction(maximum : struct Compare : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)

void SelectionSort(std::vector<int>& A, int size, int np) {
    for (int i = size - 1; i > 0; --i) {
        struct Compare max;
        max.val = A[i];
        max.index = i;

        omp_set_num_threads(np);
#pragma omp parallel for reduction(maximum:max)
        for (int j = i - 1; j >= 0; --j) {
            if (A[j] > max.val) {
                max.val = A[j];
                max.index = j;
            }
        }

        int tmp = A[i];
        A[i] = max.val;
        A[max.index] = tmp;
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
            // omp_set_num_threads(np);
            SelectionSort(A, (int)A.size(), np);

            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = stop - start;

            res += duration.count();
        }

        res /= 10;

        /*for (auto el : A)
            std::cout << el << " ";
        */
        std::cout << "\nAverage time for " << np << " is: " << res  << "\n";
    }
    return 0;
}
