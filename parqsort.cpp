#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int iterationSeq = 0;
int iterationPar = 0;

// ===============================
void generateArray(vector<int>& arr, int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100000;
    }
}
// ===============================
void printArray(const vector<int>& arr) {
    for (auto i : arr) {
        cout << i << " ";
    }
    cout << "\n";
}
// ===============================
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}
// ===============================
void sequentialQuickSort(vector<int>& arr, int low, int high, int depth = 0) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Show iteration info with array state
        iterationSeq++;
        cout << string(depth * 2, ' ') << "[Seq] Iteration " << iterationSeq
             << ": pivot index = " << pi << ", low = " << low << ", high = " << high << "\n";
        cout << string(depth * 2, ' ') << "[Seq] Array state: ";
        printArray(arr);

        sequentialQuickSort(arr, low, pi - 1, depth + 1);
        sequentialQuickSort(arr, pi + 1, high, depth + 1);
    }
}
// ===============================
void parallelQuickSort(vector<int>& arr, int low, int high, int depth = 0) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp atomic
        iterationPar++;
        cout << string(depth * 2, ' ') << "[Par] Iteration " << iterationPar
             << ": pivot index = " << pi << ", low = " << low << ", high = " << high << "\n";
        cout << string(depth * 2, ' ') << "[Par] Array state: ";
        printArray(arr);

        if (depth <= 3) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelQuickSort(arr, low, pi - 1, depth + 1);

                #pragma omp section
                parallelQuickSort(arr, pi + 1, high, depth + 1);
            }
        } else {
            sequentialQuickSort(arr, low, pi - 1, depth + 1);
            sequentialQuickSort(arr, pi + 1, high, depth + 1);
        }
    }
}
// ===============================
int main() {
    int size;
    cout << "Enter size of array: ";
    cin >> size;

    vector<int> originalArr(size), seqArr(size), parArr(size);

    int choice;
    cout << "Choose input method:\n";
    cout << "1. Manual Input\n";
    cout << "2. Random Input\n";
    cout << "Enter choice (1 or 2): ";
    cin >> choice;

    if (choice == 1) {
        cout << "Enter " << size << " elements:\n";
        for (int i = 0; i < size; i++) {
            cin >> originalArr[i];
        }
    } else if (choice == 2) {
        generateArray(originalArr, size);
    } else {
        cout << "Invalid choice. Exiting.\n";
        return 1;
    }

    // Copy arrays
    for (int i = 0; i < size; i++) {
        seqArr[i] = originalArr[i];
        parArr[i] = originalArr[i];
    }

    cout << "\nOriginal Array:\n";
    printArray(originalArr);

    cout << "\n--- Sequential Quick Sort ---\n";
    auto startTime = high_resolution_clock::now();
    sequentialQuickSort(seqArr, 0, size - 1);
    auto endTime = high_resolution_clock::now();
    auto seqTime = duration_cast<milliseconds>(endTime - startTime);
    cout << "\nFinal Sorted Array (Sequential):\n";
    printArray(seqArr);

    cout << "\n--- Parallel Quick Sort ---\n";
    startTime = high_resolution_clock::now();
    parallelQuickSort(parArr, 0, size - 1);
    endTime = high_resolution_clock::now();
    auto parTime = duration_cast<milliseconds>(endTime - startTime);
    cout << "\nFinal Sorted Array (Parallel):\n";
    printArray(parArr);

    // Print timings comparison
    cout << "\n--- Timings Comparison ---\n";
    cout << "Sequential Quick Sort Time: " << seqTime.count() << " ms\n";
    cout << "Parallel Quick Sort Time: " << parTime.count() << " ms\n";

    return 0;
}
