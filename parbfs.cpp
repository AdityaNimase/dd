// parallel_bfs.cpp
#include <iostream>
#include <vector>
#include <queue>
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

    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        cout << "Parallel BFS starting from node " << startVertex << ": ";

        vector<int> bfsOrder; // To store the BFS traversal result
        int iteration = 0;

        while (!q.empty()) {
            int size = q.size();
            iteration++;

            // Display the iteration (level of BFS)
            cout << "\nIteration " << iteration << ": ";
            vector<int> currentLevelNodes;

            #pragma omp parallel for shared(visited, q)
            for (int i = 0; i < size; ++i) {
                int current;
                
                #pragma omp critical
                {
                    if (!q.empty()) {
                        current = q.front();
                        q.pop();
                        currentLevelNodes.push_back(current); // Store the node in current level
                    }
                }

                #pragma omp parallel for
                for (int j = 0; j < adj[current].size(); ++j) {
                    int neighbor = adj[current][j];
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }

            // Print nodes at the current level (iteration)
            for (int i = 0; i < currentLevelNodes.size(); ++i) {
                cout << currentLevelNodes[i] << " ";
            }

            // Store the nodes in the final BFS order
            for (int i = 0; i < currentLevelNodes.size(); ++i) {
                bfsOrder.push_back(currentLevelNodes[i]);
            }
        }

        // Print final BFS output in a single line
        cout << "\nFinal BFS order: ";
        for (int i = 0; i < bfsOrder.size(); ++i) {
            cout << bfsOrder[i] << " ";
        }
        cout << endl;
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

    g.parallelBFS(startVertex);

    return 0;
}
