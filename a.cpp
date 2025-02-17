#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int main() {
    std::vector<int> vect = {2, 4, 6, 8, 1, 3, 5, 7};
    auto ppoint = partition_point(
        begin(vect), 
        end(vect), 
        [](int i){return i % 2 == 0;}
    );
    cout << "The first odd value is " << *ppoint << endl;
}