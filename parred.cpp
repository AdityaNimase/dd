#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>

using namespace std;

int findMin(const vector<int>& arr) {
    int minVal = numeric_limits<int>::max();
    
    #pragma omp parallel for reduction(min:minVal)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < minVal) {
            minVal = arr[i];
        }
    }
    return minVal;
}
// ===============================
int findMax(const vector<int>& arr) {
    int maxVal = numeric_limits<int>::min();
    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > maxVal) maxVal = arr[i];
    }
    return maxVal;
}
// ===============================
long long calculateSum(const vector<int>& arr) {
    long long total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < arr.size(); i++) {
        total += arr[i];
    }
    return total;
}
// ===============================
double calculateAvg(const vector<int>& arr) {
    long long total = calculateSum(arr);
    return static_cast<double>(total) / arr.size();
}
// ===============================
int main() {
    int n;

    cout << "Enter size of array: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter " << n << " integers:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int minVal = findMin(arr);
    int maxVal = findMax(arr);

    long long sum = calculateSum(arr);
    double avg = calculateAvg(arr);

    cout << "\nResults using Parallel Reduction:\n";

    cout << "\nMinimum: " << minVal;
    cout << "\nMaximum: " << maxVal;
    cout << "\nSum: " << sum;
    cout << "\nAverage: " << avg;

    return 0;
}