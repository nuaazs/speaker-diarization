## 介绍
本程序实现了K-means聚类算法，根据给定的输入参数对一组长度为192的向量进行聚类，并输出每个向量对应的聚类中心标签。

## 输入参数
- `n`：向量数量
- `k_min`：聚类中心最小个数
- `k_max`：聚类中心最大个数
- `max_iter`：最大迭代次数

## 使用方法
1. 编译源代码：
```
g++ -std=c++11  main.cpp KMeans.cpp -o kmeans
```

2. 运行可执行文件：
```
# Usage: ./kmeans <data_file> <output_file> <n> <vector_size> <k_min> <k_max> <max_iter>
./kmeans
```