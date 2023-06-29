# coding = utf-8
# @Time    : 2023-06-29  08:54:41
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: .

import torchaudio
import os
import torch
import subprocess
import numpy as np
from itertools import combinations
from flask import Flask, request, jsonify

from ECAPATDNN import emb
from utils.preprocess import save_file,save_url,vad_wav
from utils.encoder import encode_folder
from utils.encoder import similarity as sim
from utils.oss import upload_file
from utils.cmd import run_cmd
from utils.info import OutInfo
from utils.preprocess.remove_fold import remove_fold_and_file
from utils.info import OutInfo
from utils.log import logger
import cfg

from utils.log import err_logger
from utils.log import logger

app = Flask(__name__)

def find_optimal_subset(file_emb, similarity_threshold, save_wav_path=None):
    embeddings = file_emb['embedding']
    lengths = file_emb['length']
    files = list(embeddings.keys())
    # sort files by filesize (descending)
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    files= files[:cfg.MAX_WAV_NUMBER]
    # 计算所有可能的子集，并按照长度降序排序
    subsets = []
    for r in range(1, len(files) + 1):
        subsets.extend(combinations(files, r))
    subsets.sort(key=len, reverse=True)
    subsets=subsets
    
    # 逐个检查子集，找到第一个满足条件的最优解
    for subset in subsets:
        subset_embeddings = [embeddings[file] for file in subset]
        
        # 计算所有子集中的余弦相似度最小值
        min_similarity = 1.0
        similarity_list = []
        for vec1, vec2 in combinations(subset_embeddings, 2):
            similarity = sim(torch.tensor(vec1), torch.tensor(vec2))
            similarity_list.append(similarity)
            if similarity < min_similarity:
                min_similarity = similarity
        mean_similarity = np.mean(similarity_list)
        # 检查余弦相似度是否满足条件
        if mean_similarity >= similarity_threshold or len(subset) == 1:
            selected_files = list(subset)
            total_duration = sum(lengths[file] for file in selected_files)
            # 保存音频
            if save_wav_path is not None:
                os.makedirs(save_wav_path, exist_ok=True)
                selected_files = sorted(selected_files, key=lambda x: x.split("/")[-1].replace(".wav","").split("_")[0])
                selected_times = [(_data.split("/")[-1].replace(".wav","").split("_")[0],_data.split("/")[-1].replace(".wav","").split("_")[1]) for _data in selected_files]
                audio_data = np.concatenate([torchaudio.load(file)[0] for file in selected_files], axis=-1)
                torchaudio.save(os.path.join(save_wav_path,"output.wav"), torch.from_numpy(audio_data),sample_rate=8000)
            return selected_files, total_duration,os.path.join(save_wav_path,"output.wav"),selected_times
    return [], 0,os.path.join(save_wav_path,"output.wav"),selected_times


@app.route('/get_main_voice', methods=['POST'])
def get_main_voice():
    outinfo = OutInfo()
    try:
        # STEP 1: Get wav file.
        new_spkid = request.form.get("spkid","no_spkid")
        file_mode = request.form.get("file_mode","file")
        wav_url = request.form.get("wav_url","file")
        output_wav_folder = os.path.join(cfg.TEMP_PATH,new_spkid)
        os.makedirs(output_wav_folder,exist_ok=True)

        if file_mode == "file":
            new_file = request.files["wav_file"]
            if new_file.filename.split(".")[-1] not in [
                "blob", "wav", "weba", "webm", "mp3", "flac", "m4a", "ogg", "opus", "spx", "amr", "mp4", "aac",
                    "wma", "m4r", "3gp", "3g2", "caf", "aiff", "aif", "aifc", "au", "sd2", "bwf", "rf64",
            ]:
                message = f"File type error. Only support wav, weba, webm, mp3, flac, m4a, ogg, opus, spx, amr, \
                    mp4, aac, wma, m4r, 3gp, 3g2, caf, aiff, aif, aifc, au, sd2, bwf, rf64."
                err_logger.error(message)
                return outinfo.response_error(spkid=new_spkid, err_type=2, message=message)
            try:
                if "blob" in new_file.filename:
                    new_file.filename = "test.webm"
                filepath, outinfo.oss_path = save_file(
                    file=new_file, spk=new_spkid, channel=0
                )
                logger.info(f"\t\t Download success. Filepath: {filepath}")
            except Exception as e:
                remove_fold_and_file(new_spkid)
                err_logger.error(str(e))
                return outinfo.response_error(spkid=new_spkid, err_type=3, message=str(e))
        elif file_mode == "url":
            new_url = request.form.get("wav_url")
            try:
                filepath, outinfo.oss_path = save_url(
                    url=new_url, spk=new_spkid, channel=0
                )
            except Exception as e:
                remove_fold_and_file(new_spkid)
                err_logger.error(str(e))
                return outinfo.response_error(spkid=new_spkid, err_type=4, message=str(e))
        try:
            bin_path = os.path.join(output_wav_folder, os.path.basename(filepath).split(".")[0] + ".bin")
        except Exception as e:
            remove_fold_and_file(new_spkid)
            err_logger.error(str(e))
            return outinfo.response_error(spkid=new_spkid, err_type=5, message=str(e))
        try:
            vad_wav(bin_path, output_wav_folder, bin_path.replace(".bin",".txt"))
        except Exception as e:
            remove_fold_and_file(new_spkid)
            err_logger.error(str(e))
            return outinfo.response_error(spkid=new_spkid, err_type=6, message=str(e))

        try:
            embeddings = encode_folder(output_wav_folder, bin_path.replace(".bin","_encode_result.txt"))
        except Exception as e:
            remove_fold_and_file(new_spkid)
            err_logger.error(str(e))
            return outinfo.response_error(spkid=new_spkid, err_type=7, message=str(e))
        try:
            _, total_duration, wav_path, selected_times = find_optimal_subset(embeddings, 0.8, "output")
        except Exception as e:
            remove_fold_and_file(new_spkid)
            err_logger.error(str(e))
            return outinfo.response_error(spkid=new_spkid, err_type=8, message=str(e))
        try:
            url = upload_file(bucket_name="raw",filepath=wav_path,filename=f"raw_{new_spkid}.wav",save_days=180)
        except Exception as e:
            remove_fold_and_file(new_spkid)
            err_logger.error(str(e))
            return outinfo.response_error(spkid=new_spkid, err_type=9, message=str(e))
        remove_fold_and_file(new_spkid)
        response_json = {"length": total_duration, "url": url,"times":selected_times, "error": 0}
        return jsonify(response_json)
    except Exception as e:
        error_message = str(e)
        response_json = {"error": 1, "error_message": error_message}
        remove_fold_and_file(new_spkid)
        err_logger.error(error_message)
        return jsonify(response_json)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
