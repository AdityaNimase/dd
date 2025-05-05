#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Input matrix manually
void inputMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << "Enter element[" << i  << ", " << j << "] : ";
            cin >> matrix[i][j];
        }
    }
}

// Input vector manually
void inputVector(vector<int>& vec, int size) {
    for (int i = 0; i < size; i++) {
        cout << "Enter element [" << i << "]: ";
        cin >> vec[i];
    }
}

// Generate random matrix
void generateMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    srand(time(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

// Generate random vector
void generateVector(vector<int>& vec, int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 10;
    }
}

// Print a matrix
void printMatrix(const vector<vector<int>>& matrix) {
    for (auto i : matrix) {
        for (int j : i) {
            cout << j << "\t";
        }
        cout << "\n";
    }
}

// Print a vector
void printVector(const vector<int>& vec) {
    for (auto i : vec) {
        cout << i << "\t";
    }
    cout << "\n";
}

// Matrix-vector multiplication using OpenMP
vector<int> multiplyMatrixVector(const vector<vector<int>>& matrix, const vector<int>& vec, int rows, int cols) {
    vector<int> result(rows, 0);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

int main() {
    int rows, cols;
    cout << "Enter matrix dimensions (e.g., 3 4 for 3 rows and 4 columns): ";
    cin >> rows >> cols;

    vector<vector<int>> matrix(rows, vector<int>(cols));
    vector<int> vec(cols);

    int choice;

    // Choose matrix input type
    cout << "\nChoose matrix input method:\n1. Manual input\n2. Random generate\nEnter choice (1 or 2): ";
    cin >> choice;
    if (choice == 1) {
        inputMatrix(matrix, rows, cols);
    } else {
        cout << "Generating matrix...\n";
        generateMatrix(matrix, rows, cols);
    }

    // Choose vector input type
    cout << "\nChoose vector input method:\n1. Manual input\n2. Random generate\nEnter choice (1 or 2): ";
    cin >> choice;
    if (choice == 1) {
        inputVector(vec, cols);
    } else {
        cout << "Generating vector...\n";
        generateVector(vec, cols);
    }

    // Display matrix and vector
    cout << "\nMatrix:\n";
    printMatrix(matrix);

    cout << "\nVector:\n";
    printVector(vec);

    // Perform multiplication
    auto startTime = high_resolution_clock::now();
    vector<int> result = multiplyMatrixVector(matrix, vec, rows, cols);
    auto endTime = high_resolution_clock::now();

    auto total = duration_cast<milliseconds>(endTime - startTime);

    cout << "\nResult Vector:\n";
    printVector(result);

    cout << "\nTime taken to calculate: " << total.count() << " ms\n";

    return 0;
}
