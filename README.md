# 主叫音频分离
[接口文档](./API.md) | [聚类算法](./clustering.md) 

## 介绍
用于提取合轨后的多说话人音频中的主要说话人，用于后续声纹特征编码。

## 启动方式
1. 修改配置文件
```shell
vim flask_backend/cfg.py
```

2. 启动服务
```shell
cd flask_backend
python server.py
```

3. 接口测试
```shell
python test_api.py
```