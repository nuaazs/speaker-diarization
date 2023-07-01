import requests
import wget
import os

ROOT = "/home/zhaosheng/speaker-diarization/data/speaker_diarisation_test_data"
files = [os.path.join(ROOT,_file) for _file in os.listdir(ROOT) if _file.endswith(".wav")]

for wav_file_path in files:
    #wav_file_path = "/home/zhaosheng/speaker-diarization/data/speaker_diarisation_test_data/2a28454a9ee6857334ef28a463cece3b-江苏-常州-2023年01月13日20时55分30秒-13584539654-1673614514.430617488000.wav"
    spkid = wav_file_path.split("/")[-1].split(".")[0].split("-")[-2]
    url = "http://localhost:5001/get_main_voice"
    payload = {
        "file_mode": "file",
        "spkid":spkid,
    }
    files=[
    ('wav_file',(wav_file_path.split("/")[-1],open(wav_file_path,'rb'),'application/octet-stream'))
    ]
    response = requests.post(url, data=payload, files=files)
    print(response.json())
    url = response.json()["url"]
    wget.download(url, out=f"output/{spkid}.wav")

