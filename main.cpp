#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "KMeans.h"

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 9) {
        std::cout << "Usage: ./kmeans <data_file> <output_file> <n> <vector_size> <k_min> <k_max> <max_iter> <discard_threshold> <calc_type>" << std::endl;
        return 1;
    }

    // 从命令行参数获取数据文件路径、输出文件路径、数据点个数、向量长度、k_min、k_max、max_iter、丢弃阈值和计算类型
    std::string dataFile = argv[1];
    std::string outputFile = argv[2];
    int n = std::stoi(argv[3]);
    int vectorSize = std::stoi(argv[4]);
    int k_min = std::stoi(argv[5]);
    int k_max = std::stoi(argv[6]);
    int max_iter = std::stoi(argv[7]);
    double discard_threshold = std::stod(argv[8]);
    KMeans::DistanceType distance_type;

    if (std::string(argv[9]) == "dist") {
        distance_type = KMeans::DistanceType::EUCLIDEAN;
    } else if (std::string(argv[9]) == "cos") {
        distance_type = KMeans::DistanceType::COSINE;
    } else {
        std::cerr << "Invalid calc_type. Must be either 'dist' or 'cos'." << std::endl;
        return 1;
    }

    // 读取二进制文件中的数据
    std::ifstream file(dataFile, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open data file." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> data(n, std::vector<double>(vectorSize));

    for (int i = 0; i < n; ++i) {
        if (!file.read(reinterpret_cast<char*>(data[i].data()), vectorSize * sizeof(float))) {
            std::cerr << "Failed to read data from file." << std::endl;
            return 1;
        }
    }

    file.close();

    // 运行K-means聚类算法
    std::vector<int> labels = KMeans::run(data, n, k_min, k_max, max_iter, discard_threshold, distance_type);

    // 将聚类结果写入输出文件
    std::ofstream output(outputFile);
    if (!output) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        output << i << "," << labels[i] << std::endl;
    }

    output.close();

    return 0;
}
