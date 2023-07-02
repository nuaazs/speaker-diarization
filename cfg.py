# coding = utf-8
# @Time    : 2022-09-05  09:43:48
# @Author  : zhaosheng@nuaa.edu.cn

WORKERS = 4
SR = 16000
WAV_START=0
WAV_LENGTH=999
TIME_TH = 2
DEVICE = "cuda:0"
THREADS = 10
TEST_THREADS = 20
WORKER_CONNECTIONS = 1000
PORT = 7777
VAD_BIN="/home/zhaosheng/speaker-diarization/bin/vad"
MAX_WAV_NUMBER = 10
ENCODE_MODEL_LIST = ["ECAPATDNN"]#, "CAMPP"]  #  ,"CAMPP"
BLACK_TH = {"ECAPATDNN": 0.78, "CAMPP": 0.78}
EMBEDDING_LEN = {"ECAPATDNN": 192, "CAMPP": 512}
MODEL_PATH = {"ECAPATDNN":"/VAF/src/nn/ECAPATDNN","CAMPP":"/VAF/src/nn/CAMPP"}
TEMP_PATH = "/tmp"
damo="/home/zhaosheng/asr_damo_websocket/online/damo"
#######################################################
####################  Databases #######################
#######################################################
MYSQL = {
    "host": "192.168.3.169",
    "port": 3306,
    "db": "si",
    "username": "zhaosheng",
    "passwd": "Nt3380518",
}

REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 1,
    "test_db": 2,
    "password": "",
}

MINIO = {
    "host": "192.168.3.169",
    "port": 9000,
    "access_key": "zhaosheng",
    "secret_key": "zhaosheng",
    "test_save_days": 30,
    "register_save_days": -1,
    "register_raw_bucket": "register_raw",
    "register_preprocess_bucket": "register_preprocess",
    "test_raw_bucket": "test_raw",
    "test_preprocess_bucket": "test_preprocess",
    "pool_raw_bucket": "pool_raw",
    "pool_preprocess_bucket": "pool_preprocess",
    "black_raw_bucket": "black_raw",
    "black_preprocess_bucket": "black_preprocess",
    "white_raw_bucket": "white_raw",
    "white_preprocess_bucket": "white_preprocess",
    "zs": b"zhaoshengzhaoshengnuaazs",
}
BUCKETS = ["raw", "preprocess", "preprocessed", "testing", "sep"]