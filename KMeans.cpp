#include "KMeans.h"
#include <cmath>
#include <limits>
#include <random>

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

    // K-means 聚类
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<int> clusterSizes(k_min, 0);
        std::vector<std::vector<double>> newCentroids(k_min, std::vector<double>(data[0].size(), 0.0));

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
        for (int i = 0; i < k_min; ++i) {
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
            double distance = 0.0;

            if (distance_type == DistanceType::EUCLIDEAN) {
                // 计算欧氏距离
                for (int j = 0; j < point.size(); ++j) {
                    double diff = point[j] - centroids[nearestCluster][j];
                    distance += diff * diff;
                }

                distance = std::sqrt(distance);
            } else if (distance_type == DistanceType::COSINE) {
                // 计算余弦相似度
                double dotProduct = 0.0;
                double normA = 0.0;
                double normB = 0.0;

                for (int j = 0; j < point.size(); ++j) {
                    dotProduct += point[j] * centroids[nearestCluster][j];
                    normA += point[j] * point[j];
                    normB += centroids[nearestCluster][j] * centroids[nearestCluster][j];
                }

                distance = 1.0 - dotProduct / (std::sqrt(normA) * std::sqrt(normB));
            }

            if (distance > discard_threshold) {
                labels[i] = -1;
            }
        }
    }

    return labels;
}

int KMeans::findNearestCluster(const std::vector<double>& v, const std::vector<std::vector<double>>& centroids, DistanceType distance_type) {
    int nearestCluster = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < centroids.size(); ++i) {
        const std::vector<double>& centroid = centroids[i];

        double distance = 0.0;

        if (distance_type == DistanceType::EUCLIDEAN) {
            // 计算欧氏距离
            for (int j = 0; j < v.size(); ++j) {
                double diff = v[j] - centroid[j];
                distance += diff * diff;
            }

            distance = std::sqrt(distance);
        } else if (distance_type == DistanceType::COSINE) {
            // 计算余弦相似度
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;

            for (int j = 0; j < v.size(); ++j) {
                dotProduct += v[j] * centroid[j];
                normA += v[j] * v[j];
                normB += centroid[j] * centroid[j];
            }

            distance = 1.0 - dotProduct / (std::sqrt(normA) * std::sqrt(normB));
        }

        if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = i;
        }
    }

    return nearestCluster;
}
