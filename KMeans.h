#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

class KMeans {
public:
    std::vector<int> run(const std::vector<std::vector<double> >& data, int k_min, int k_max, int max_iter);
private:
    double distance(const std::vector<double>& v1, const std::vector<std::vector<double> >& centroids);
    int findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double> >& centroids);
};

#endif  // KMEANS_H
