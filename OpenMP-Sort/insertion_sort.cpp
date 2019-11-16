#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <map>

void InsertionSort(std::vector<int>& A, int left, int right, int np) {
    if (left == right)
        return;

    int inp = A[right];
    int rright = right - 1;
    InsertionSort(A, left, rright, np);

    if (inp >= A[rright])
        return;

    int ind = 0;
    omp_set_num_threads(np);
#pragma omp parallel for shared(ind)
    for (int i = 0; i < rright; ++i) {
        if (inp >= A[i] && inp <= A[i+1]) {
            ind = i + 1;
        }
    }
    auto tmp = A[rright];
    for (int i = rright; i > ind; --i) {
        A[i] = A[i-1];
    }
    A[ind] = inp;
    A[right] = tmp;
}
int main() {
    int N = 1000000, a = 0;
    std::vector<int> A, B;
    A.reserve(N);

    while (std::cin >> a) {
        A.push_back(a);
        B.push_back(a);
    }

    std::vector<int> sizes = {1, 2, 4, 8, 16};

    int cnt = 3;

    for (auto np : sizes) {
        double res = 0;

        for (int i = 0; i < cnt; ++i) {
            A = B;
            auto start = std::chrono::high_resolution_clock::now();

            InsertionSort(A, 0, (int)(A.size() - 1), np);

            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = stop - start;

            res += duration.count();
        }

        res /= cnt;

        /*for (auto el : A)
            std::cout << el << " ";*/

        std::cout << "\nAverage time for " << np << " is: " << res  << "\n";
    }
    return 0;
}
