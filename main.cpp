#include <iostream>
#include <vector>
#include <random>
#include "KMeans.h"

int main() {
    // 生成测试用例数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    int n = 100; // 数据点个数
    int vectorSize = 192; // 向量长度
    std::vector<std::vector<double>> data(n, std::vector<double>(vectorSize));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < vectorSize; ++j) {
            data[i][j] = dis(gen);
        }
    }

    // 设置聚类中心个数范围和最大迭代次数
    int k_min = 4;
    int k_max = 5;
    int max_iter = 100;

    // 运行K-means聚类算法
    std::vector<int> labels = KMeans::run(data, n, k_min, k_max, max_iter);

    // 输出聚类中心的标签
    std::cout << "Cluster labels:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << labels[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}
