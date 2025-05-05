#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void inputMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << "Enter element[" << i << ", " << j << "] : ";
            cin >> matrix[i][j];
        }
    }
}

void generateMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            cout << val << "\t";
        }
        cout << "\n";
    }
}

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& a, const vector<vector<int>>& b, int r, int c1, int c2) {
    vector<vector<int>> result(r, vector<int>(c2, 0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

int main() {
    srand(time(0));

    int rowsA, colsA;
    cout << "Enter dimensions of Matrix A (rows x cols): ";
    cin >> rowsA >> colsA;

    int rowsB, colsB;
    cout << "Enter dimensions of Matrix B (rows x cols): ";
    cin >> rowsB >> colsB;

    // Validate matrix multiplication condition
    if (colsA != rowsB) {
        cout << "\nMatrix multiplication not possible!\n";
        cout << "Reason: Columns of Matrix A (" << colsA << ") != Rows of Matrix B (" << rowsB << ")\n";
        return 1;
    }

    vector<vector<int>> matrix_a(rowsA, vector<int>(colsA));
    vector<vector<int>> matrix_b(rowsB, vector<int>(colsB));

    int choice;

    // Matrix A input
    cout << "\nMatrix A:\n";
    cout << "Choose input method for Matrix A:\n1. Manual Input\n2. Random Generation\nEnter choice: ";
    cin >> choice;
    if (choice == 2) {
        generateMatrix(matrix_a, rowsA, colsA);
    } else {
        inputMatrix(matrix_a, rowsA, colsA);
    }

    cout << "\nMatrix A values:\n";
    printMatrix(matrix_a);

    // Matrix B input
    cout << "\nMatrix B:\n";
    cout << "Choose input method for Matrix B:\n1. Manual Input\n2. Random Generation\nEnter choice: ";
    cin >> choice;
    if (choice == 2) {
        generateMatrix(matrix_b, rowsB, colsB);
    } else {
        inputMatrix(matrix_b, rowsB, colsB);
    }

    cout << "\nMatrix B values:\n";
    printMatrix(matrix_b);

    // Perform matrix multiplication
    auto startTime = high_resolution_clock::now();
    vector<vector<int>> result = multiplyMatrices(matrix_a, matrix_b, rowsA, colsA, colsB);
    auto endTime = high_resolution_clock::now();

    auto total = duration_cast<milliseconds>(endTime - startTime);

    cout << "\nResult Matrix:\n";
    printMatrix(result);

    double time_ms = total.count();
    string unit = "ms";

    if (time_ms > 1000.0) {
        time_ms = time_ms / 1000.0;
        unit = "s";
    }

    cout << "Time taken to calculate: " << time_ms << " " << unit << "\n";

    return 0;
}
