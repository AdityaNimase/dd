#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to generate a random array
void generateArray(vector<int>& arr, int size) {
    srand(time(0)); // seed the random number generator
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100000; // random numbers between 0 and 99999
    }
}

// Function to take user input for array
void inputArray(vector<int>& arr, int size) {
    cout << "Enter the elements of the array:\n";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }
}

// Function to display array
void printArray(const vector<int>& arr) {
    for (auto i: arr) {
        cout << i << " ";
    }
    cout << "\n";
}

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
        cout << "Iteration " << i + 1 << " (Sequential): ";
        printArray(arr);
    }
}

// Parallel Bubble Sort using OpenMP (Odd-Even Transposition Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n; i++) {
        // Even Phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }

        // Odd Phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
        
        cout << "Iteration " << i + 1 << " (Parallel): ";
        printArray(arr);
    }
}

void copyArray(const vector<int>& source, vector<int>& dest) {
    for (int i = 0; i < source.size(); i++) {
        dest[i] = source[i];
    }
}

int main(int argc, char const *argv[]) {
    int size;
    cout << "Enter size of array: ";
    cin >> size;

    vector<int> originalArr(size), seqArr(size), parArr(size);

    int choice;
    cout << "Choose array type:\n";
    cout << "1. Manually input array\n";
    cout << "2. Randomly generate array\n";
    cin >> choice;

    if (choice == 1) {
        inputArray(originalArr, size);
    } else if (choice == 2) {
        generateArray(originalArr, size);
    } else {
        cout << "Invalid choice! Exiting program.\n";
        return 1;
    }

    copyArray(originalArr, seqArr);
    copyArray(originalArr, parArr);

    cout << "\nOriginal Array: ";
    printArray(originalArr);

    // Sequential Bubble Sort
    cout << "Calculating sequential time...\n";
    auto startTime = high_resolution_clock::now();
    sequentialBubbleSort(seqArr);
    auto endTime = high_resolution_clock::now();
    auto seqTime = duration_cast<milliseconds>(endTime - startTime);

    // Parallel Bubble Sort
    cout << "Calculating parallel time...\n";
    startTime = high_resolution_clock::now();
    parallelBubbleSort(parArr);
    endTime = high_resolution_clock::now();
    auto parTime = duration_cast<milliseconds>(endTime - startTime);

    cout << "\nSorted Array (Sequential): ";
    printArray(seqArr);

    cout << "Sorted Array (Parallel): ";
    printArray(parArr);

    cout << "Performance\n";
    cout << "Sequential Time: " << seqTime.count() << " milliseconds\n";
    cout << "Parallel Time: " << parTime.count() << " milliseconds\n";

    return 0;
}
