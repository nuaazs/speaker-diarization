# utils

import cfg
import os
import importlib
import torchaudio
import numpy as np
ENCODE_MODEL_LIST = cfg.ENCODE_MODEL_LIST
from utils.log import logger

emb_dict = {}
for model in ENCODE_MODEL_LIST:
    module = importlib.import_module(f"utils.encoder.{model}")
    emb_dict[model] = module.emb
# from utils.encoder.ECAPATDNN import emb as ECAPATDNN_emb
# from utils.encoder.CAMPP import emb as CAMPP_emb
# 打印 emb_dict 占用内存大小
import sys


def encode_folder(folder_path, txt_save_path):
    file_emb = {}
    file_emb["embedding"] = {}
    file_emb["length"] = {}
    wav_files = sorted([
        os.path.join(folder_path, _file)
        for _file in os.listdir(folder_path)
        if _file.endswith(".wav") and "raw" not in _file
    ])
    with open(txt_save_path, "w") as f:
        i = 0
        for wav_file in wav_files:
            _data = torchaudio.load(wav_file)[0].reshape(1, -1)
            if _data.shape[1] < cfg.SR * cfg.TIME_TH:
                continue
            embeddings = emb_dict["ECAPATDNN"].encode_batch(_data)
            embeddings = embeddings.detach().cpu().numpy().reshape(-1)
            embeddings = embeddings.astype(np.float32)
            file_emb["embedding"][wav_file] = embeddings
            file_emb["length"][wav_file] = _data.shape[1] / 8000
            f.write(f"{wav_file},{i}\n")
            i += 1
    return file_emb
