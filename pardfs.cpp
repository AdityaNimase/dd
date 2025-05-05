// parallel_dfs.cpp
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // For undirected graph or tree
    }

    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);

        cout << "Parallel DFS starting from node " << startVertex << ": ";

        vector<int> dfsOrder; // To store the DFS traversal result
        int iteration = 0;

        // Call DFS Utility Function
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                DFSUtil(startVertex, visited, iteration, dfsOrder);
            }
        }

        // Print final DFS output in a single line
        cout << "\nFinal DFS order: ";
        for (int i = 0; i < dfsOrder.size(); ++i) {
            cout << dfsOrder[i] << " ";
        }
        cout << endl;
    }

    void DFSUtil(int u, vector<bool>& visited, int& iteration, vector<int>& dfsOrder) {
        visited[u] = true;
        iteration++;
        cout << "\nIteration " << iteration << ": " << u;

        dfsOrder.push_back(u); // Store the visited node in DFS order

        #pragma omp parallel for
        for (int i = 0; i < adj[u].size(); ++i) {
            int v = adj[u][i];
            if (!visited[v]) {
                DFSUtil(v, visited, iteration, dfsOrder);
            }
        }
    }
};

int main() {
    int V, E;

    cout << "Enter number of vertices: ";
    cin >> V;
    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;
    cout << "Enter edges (u v): \n";
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int startVertex;
    cout << "Enter the start vertex: ";
    cin >> startVertex;

    g.parallelDFS(startVertex);

    return 0;
}
