#include <iostream>
#include "KMeans.h"

int main() {
    // 输入参数
    int n; // 向量数量
    int k_min; // 聚类中心最小个数
    int k_max; // 聚类中心最大个数
    int max_iter; // 最大迭代次数

    // 根据需求获取向量数据，并保存到data向量中
    std::vector<std::vector<double> > data;

    // 实例化KMeans对象
    KMeans kmeans;

    // 调用run函数进行聚类
    std::vector<int> clusterLabels = kmeans.run(data, k_min, k_max, max_iter);

    // 输出参数，即n个聚类中心的标签
    for (int i = 0; i < clusterLabels.size(); ++i) {
        std::cout << "Vector " << i << ": Cluster " << clusterLabels[i] << std::endl;
    }

    return 0;
}
