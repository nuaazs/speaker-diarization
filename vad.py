import subprocess
import os
from itertools import combinations
import torchaudio
import torch
import numpy as np

from utils.encoder import similarity as sim
from utils.encoder import encode_folder

def vad(filepath,save_folder_path,smooth_threshold=0.5,min_duration=2):
    os.makedirs(save_folder_path,exist_ok=True)
    bin_path = f"{filepath.split('/')[-1][:-4]}.bin"
    bin_path = os.path.join(save_folder_path,bin_path)
    cmd = f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  {bin_path}> /dev/null 2>&1"
    subprocess.call(cmd,shell=True)
    vad_cmd = f"vad {bin_path} {save_folder_path} {save_folder_path}/result.txt {smooth_threshold} {min_duration}"
    subprocess.call(vad_cmd,shell=True)
    return


def find_optimal_subset(file_emb, similarity_threshold, save_wav_path=None,max_wav_number=10):
    embeddings = file_emb['embedding']
    lengths = file_emb['length']
    files = list(embeddings.keys())
    # sort files by filesize (descending)
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    files= files[:max_wav_number]
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

if __name__ == '__main__':
    wav_file = "/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav"
    save_folder_path = "./test_output"
    vad(wav_file,save_folder_path)
    embeddings = encode_folder(save_folder_path, os.path.join(save_folder_path,"encode_result.txt"))
    result = find_optimal_subset(embeddings, similarity_threshold=0.8, save_wav_path=os.path.join(save_folder_path,"final_.txt"),max_wav_number=10)
    print(result)