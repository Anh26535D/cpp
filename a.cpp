#include <iostream>
#include <string>

using namespace std;

class B {
private:
    int b;
public:
    B() {}
    B(int i) : b(i) {}
    int show() { return b; }
};

class C {
private:    
    B b;
public:
    C(int i) { b = B(i); }
    friend void show();
};

void show() {
    C c(10);
    cout << "value of b is: " << c.b.show() << endl;
}

int main() {
    show();
    return 0;
}