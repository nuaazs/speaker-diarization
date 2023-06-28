#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

class KMeans {
public:
    static std::vector<int> run(const std::vector<std::vector<double>>& data, int n, int k_min, int k_max, int max_iter);

private:
    static int findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double>>& centroids);
};

#endif
