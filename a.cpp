#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <execution>

using namespace std;

constexpr int kMaxIters = 1000;
constexpr int kMaxSize = 100000000;

void bm_innerproduct() {
    auto start = chrono::steady_clock::now();
    vector<int> v1(kMaxSize);
    vector<int> v2(kMaxSize);
    for (int i = 0; i < kMaxIters; ++i) {
        int result = transform_reduce(
            begin(v1), end(v1), begin(v2), 0
        );
    }
    auto end = chrono::steady_clock::now();
    cout << "inner_product: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
}

void bm_partransformreduce() {
    auto start = chrono::steady_clock::now();
    vector<int> v1(kMaxSize);
    vector<int> v2(kMaxSize);
    for (int i = 0; i < kMaxIters; ++i) {
        int result = transform_reduce(
            execution::par_unseq,
            begin(v1), end(v1), begin(v2), 0
        );
    }
    auto end = chrono::steady_clock::now();
    cout << "par_transform_reduce: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
}

int main() {
    bm_innerproduct();
    bm_partransformreduce();
    return 0;
}
