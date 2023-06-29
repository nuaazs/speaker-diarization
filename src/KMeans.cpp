#include "../include/KMeans.h"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>

std::vector<int> KMeans::run(const std::vector<std::vector<double>>& data, int n, int k_min, int k_max, int max_iter, double discard_threshold, DistanceType distance_type) {
    std::vector<int> labels(n, -1);
    std::vector<std::vector<double>> centroids(k_min);

    // 从数据点中随机选择初始质心
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, n - 1);

    for (int i = 0; i < k_min; ++i) {
        centroids[i] = data[dis(gen)];
    }

    int bestK = k_min;         // 最优的聚类中心数目
    double minAvgDistance = std::numeric_limits<double>::max();   // 平均距离的最小值

    // 遍历聚类中心数目
    for (int k = k_min; k <= k_max; k++) {
        std::vector<int> clusterSizes(k, 0);
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(data[0].size(), 0.0));

        // K-means 聚类
        for (int iter = 0; iter < max_iter; ++iter) {
            // 将每个数据点分配到最近的质心
            for (int i = 0; i < n; ++i) {
                int nearestCluster = findNearestCluster(data[i], centroids, distance_type);
                labels[i] = nearestCluster;
                clusterSizes[nearestCluster]++;
                for (int j = 0; j < data[i].size(); ++j) {
                    newCentroids[nearestCluster][j] += data[i][j];
                }
            }

            // 更新质心
            for (int i = 0; i < k; ++i) {
                if (clusterSizes[i] > 0) {
                    for (int j = 0; j < centroids[i].size(); ++j) {
                        centroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                    }
                }
            }

            // 清空 newCentroids 和 clusterSizes
            newCentroids.assign(k, std::vector<double>(data[0].size(), 0.0));
            clusterSizes.assign(k, 0);
        }

        // 计算平均距离
        double totalDistance = 0.0;
        for (int i = 0; i < n; ++i) {
            totalDistance += calculateDistance(data[i], centroids[labels[i]], distance_type);
        }
        double avgDistance = totalDistance / n;

        // 更新最优的聚类中心数目和平均距离
        if (avgDistance < minAvgDistance) {
            minAvgDistance = avgDistance;
            bestK = k;
        }
    }

    // 重新初始化随机数生成器
    gen.seed(rd());

    // 更新聚类中心数目为最优值
    centroids.resize(bestK);

    // 从数据点中随机选择初始质心
    for (int i = 0; i < bestK; ++i) {
        centroids[i] = data[dis(gen)];
    }

    // K-means 聚类
    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<int> clusterSizes(bestK, 0);
        std::vector<std::vector<double>> newCentroids(bestK, std::vector<double>(data[0].size(), 0.0));

        // 将每个数据点分配到最近的质心
        for (int i = 0; i < n; ++i) {
            int nearestCluster = findNearestCluster(data[i], centroids, distance_type);
            labels[i] = nearestCluster;
            clusterSizes[nearestCluster]++;
            for (int j = 0; j < data[i].size(); ++j) {
                newCentroids[nearestCluster][j] += data[i][j];
            }
        }

        // 更新质心
        for (int i = 0; i < bestK; ++i) {
            if (clusterSizes[i] > 0) {
                for (int j = 0; j < centroids[i].size(); ++j) {
                    centroids[i][j] = newCentroids[i][j] / clusterSizes[i];
                }
            }
        }
    }

    // 丢弃难以分类的数据点
    if (discard_threshold > 0.0) {
        for (int i = 0; i < n; ++i) {
            std::vector<double> point = data[i];
            int nearestCluster = findNearestCluster(point, centroids, distance_type);
            double distance = calculateDistance(point, centroids[nearestCluster], distance_type);

            if (distance > discard_threshold) {
                labels[i] = -1;
            }
        }
    }

    return labels;
}

double KMeans::calculateDistance(const std::vector<double>& v1, const std::vector<double>& v2, DistanceType distance_type) {
    double distance = 0.0;

    if (distance_type == DistanceType::EUCLIDEAN) {
        // 计算欧氏距离
        for (int j = 0; j < v1.size(); ++j) {
            double diff = v1[j] - v2[j];
            distance += diff * diff;
        }

        distance = std::sqrt(distance);
    } else if (distance_type == DistanceType::COSINE) {
        // 计算余弦相似度
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int j = 0; j < v1.size(); ++j) {
            dotProduct += v1[j] * v2[j];
            normA += v1[j] * v1[j];
            normB += v2[j] * v2[j];
        }

        distance = 1.0 - dotProduct / (std::sqrt(normA) * std::sqrt(normB));
    } else {
        // 处理其他距离类型
        // TODO: 添加适当的代码来处理其他距离类型
    }

    return distance;
}

int KMeans::findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double>>& centroids, DistanceType distance_type) {
    int nearestCluster = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < centroids.size(); ++i) {
        const std::vector<double>& centroid = centroids[i];

        double distance = calculateDistance(v, centroid, distance_type);

        if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = i;
        }
    }

    return nearestCluster;
}
