#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int sequentialIteration = 0;
int parallelIteration = 0;

// Function to generate random array
void generateArray(vector<int>& arr, int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100000;  // 0 to 99,999
    }
}

// Function to print array
void printArray(const vector<int>& arr) {
    for (auto i : arr) {
        cout << i << " ";
    }
    cout << "\n";
}

// Merge function for sequential
void mergeSequential(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int l = 0; l < temp.size(); l++) {
        arr[left + l] = temp[l];
    }

    cout << "Iteration " << ++sequentialIteration << " (Sequential Merge of [" << left << ", " << right << "]): ";
    printArray(arr);
}

// Merge function for parallel
void mergeParallel(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int l = 0; l < temp.size(); l++) {
        arr[left + l] = temp[l];
    }

    cout << "Iteration " << ++parallelIteration << " (Parallel Merge of [" << left << ", " << right << "]): ";
    printArray(arr);
}

// Sequential Merge Sort
void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = (left + right) / 2;

    mergeSortSequential(arr, left, mid);
    mergeSortSequential(arr, mid + 1, right);

    mergeSequential(arr, left, mid, right);
}

// Parallel Merge Sort
void mergeSortParallel(vector<int>& arr, int left, int right, int depth = 0) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    if (depth <= 3) {
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, left, mid, depth + 1);

            #pragma omp section
            mergeSortParallel(arr, mid + 1, right, depth + 1);
        }
    } else {
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
    }

    mergeParallel(arr, left, mid, right);
}

// Manual copy
void copyArray(const vector<int>& source, vector<int>& dest) {
    for (int i = 0; i < source.size(); ++i) {
        dest[i] = source[i];
    }
}

int main() {
    int size, choice;

    cout << "Enter size of array: ";
    cin >> size;

    vector<int> original(size), seqArr(size), parArr(size);

    cout << "Choose input method:\n1. Manual input\n2. Random values\nEnter choice: ";
    cin >> choice;

    if (choice == 2) {
        generateArray(original, size);
    } else {
        cout << "Enter " << size << " elements:\n";
        for (int i = 0; i < size; ++i) {
            cin >> original[i];
        }
    }

    copyArray(original, seqArr);
    copyArray(original, parArr);

    cout << "\nOriginal Array:\n";
    printArray(original);

    // Sequential Sort
    cout << "\n--- Sequential Merge Sort ---\n";
    auto startTime = high_resolution_clock::now();
    mergeSortSequential(seqArr, 0, size - 1);
    auto endTime = high_resolution_clock::now();
    auto seqTime = duration_cast<milliseconds>(endTime - startTime);

    // Parallel Sort
    cout << "\n--- Parallel Merge Sort ---\n";
    startTime = high_resolution_clock::now();
    mergeSortParallel(parArr, 0, size - 1);
    endTime = high_resolution_clock::now();
    auto parTime = duration_cast<milliseconds>(endTime - startTime);

    cout << "\nSorted Array (Sequential):\n";
    printArray(seqArr);

    cout << "\nSorted Array (Parallel):\n";
    printArray(parArr);

    cout << "\nPerformance:\n";
    cout << "Sequential Time: " << seqTime.count() << " ms\n";
    cout << "Parallel Time: " << parTime.count() << " ms\n";

    return 0;
}
