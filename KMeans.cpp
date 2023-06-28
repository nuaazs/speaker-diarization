#include "KMeans.h"
#include <cmath>
#include <limits>
#include <random>

int KMeans::findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double>>& centroids) {
    int nearestCluster = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < centroids.size(); ++i) {
        const std::vector<double>& centroid = centroids[i];

        // 计算数据点与质心之间的欧氏距离
        double distance = 0.0;
        for (int j = 0; j < v.size(); ++j) {
            double diff = v[j] - centroid[j];
            distance += diff * diff;
        }

        if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = i;
        }
    }

    return nearestCluster;
}

std::vector<int> KMeans::run(const std::vector<std::vector<double>>& data, int k_min, int k_max, int max_iter) {
    std::vector<int> labels(data.size(), -1);
    std::vector<std::vector<double>> centroids(k_min);

    // 从数据点中随机选择初始质心
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, data.size() - 1);

    for (int i = 0; i < k_min; ++i) {
        centroids[i] = data[dis(gen)];
    }

    // K-means 聚类
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<int> clusterSizes(k_min, 0);
        std::vector<std::vector<double>> newCentroids(k_min, std::vector<double>(data[0].size(), 0.0));

        // 将每个数据点分配到最近的质心
        for (int i = 0; i < data.size(); ++i) {
            int nearestCluster = findNearestCluster(data[i], centroids);
            labels[i] = nearestCluster;
            clusterSizes[nearestCluster]++;
            for (int j = 0; j < data[i].size(); ++j) {
                newCentroids[nearestCluster][j] += data[i][j];
            }
        }

        // 更新质心
        for (int i = 0; i < k_min; ++i) {
            if (clusterSizes[i] > 0) {
                for (int j = 0; j < centroids[i].size(); ++j) {
                    centroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                }
            }
        }
    }

    return labels;
}
