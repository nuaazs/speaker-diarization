from utils.vad import nn_vad, energybase_vad
from utils.encoder import encode_folder
from utils.cluster import find_optimal_subset
from utils.mandarin import mandarin_filter
from utils.oss.upload import upload_file, upload_files, remove_urls_from_bucket
from asr import file_upload_offline as asr_offline
from utils.nlp import classify_text
import subprocess
import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def get_spectrogram(audio_file_path,save_png_path):
    # 加载音频文件
    audio, sr = librosa.load(audio_file_path)

    # 计算短时傅里叶变换（STFT）
    stft = librosa.stft(audio)

    # 转换为分贝刻度
    spectrogram = librosa.amplitude_to_db(np.abs(stft))

    # 绘制频谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    # 将图表转换为图像数组
    fig = plt.gcf()
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.savefig(save_png_path)

    return

def create_audio_players(audio_files):
    audio_players = []
    for _index,audio_file in enumerate(audio_files,1):
        
        # audio file is local file path on server
        # 居中
        audio_player = f"<audio style='width: 50%; margin: 0 auto;' controls><source src='{audio_file}' type='audio/wav'> </audio>"
        # show index of audio file in html
        audio_player += f"<p>第{_index}个片段，{audio_file}：</p>"
        audio_players.append(audio_player)
    return audio_players


def generate_html_content(audio_players, title):
    # 居中标题 
    html_content = f"<h2 style='text-align:center'>{title}</h2>"
    
    # 居中div内容
    html_content += "<div style='text-align:center'>"
    for audio_player in audio_players:
        html_content += audio_player
    html_content += "</div>"
    
    # # 添加折叠/展开按钮
    # html_content += """
    # <script>
    # function toggleContent() {
    #     var content = document.getElementById('content');
    #     if (content.style.display === 'none') {
    #         content.style.display = 'block';
    #     } else {
    #         content.style.display = 'none';
    #     }
    # }
    # </script>
    # <button onclick='toggleContent()' style='display:block; margin: 0 auto;'>折叠/展开</button>
    # """
    
    # # 添加初始状态为折叠
    # html_content += "<style>#content { display: none; }</style>"
    
    # # 将div包装在一个可折叠的容器中
    # html_content = f"<div id='content'>{html_content}</div>"
    
    return html_content

def pipeline(filepath, vad_type, spkid):
    # return example
    tmp_folder = f"/tmp/{spkid}"
    # 步骤一： VAD
    if vad_type == 'nn_vad':
        wav_files = nn_vad(filepath, tmp_folder)
    elif vad_type == 'energybase_vad':
        wav_files = energybase_vad(filepath, tmp_folder)
    else:
        return None
    vad_urls = upload_files("testing", wav_files, save_days=180)
    vad_result =  [f"文件: {wav}" for wav in wav_files]
    vad_audios = wav_files
    
    print(f"\t * -> VAD结果: {vad_result}")
    
    # 步骤二： 普通话过滤
    mandarin_wavs = mandarin_filter(wav_files)
    mandarin_urls = upload_files("testing", mandarin_wavs, save_days=180)
    mandarin_result = [f"文件: {wav}" for wav in mandarin_wavs]
    mandarin_audios = mandarin_wavs
    print(f"\t * -> 普通话过滤结果: {mandarin_result}")

    # 步骤三： 提取特征
    file_emb = encode_folder(mandarin_wavs)
    print(f"\t * -> 提取特征结果: {file_emb}")

    # 步骤四： 聚类
    selected_files, total_duration, url, wav_file_path, selected_times = find_optimal_subset(file_emb, spkid=spkid, similarity_threshold=0.8, save_wav_path=tmp_folder)
    print(f"\t * -> 聚类结果: {selected_times}")
    print(f"\t * -> 聚类结果URL: {url}")
    selected_urls = upload_files("testing", selected_files, save_days=180)

    # 步骤五： ASR
    text = asr_offline(filepath)
    print(f"\t * -> ASR结果: {text}")

    # 步骤六： NLP
    nlp_result = classify_text(text)
    print(f"\t * -> 文本分类结果: {nlp_result}")

    subprocess.call(f"rm -rf {tmp_folder}", shell=True)
    return {
        "raw_file_path": filepath,
        "vad_result": vad_result,
        "vad_urls" : vad_urls,
        "play_vad_files": vad_audios,
        "mandarin_filter_result": mandarin_result,
        "mandarin_urls":mandarin_urls,
        "play_mandarin_files": mandarin_audios,
        "embeddings": file_emb,
        "selected_times": selected_times,
        "url": url,
        "asr_result": text,
        "nlp_result": nlp_result,
        "total_duration": total_duration,
        "selected_urls": selected_urls,

        
    }

def process_audio(input_file, vad_type):
    output = {}
    output["file_name"] = input_file.name
    output["audio_length"] = "待计算"
    # output["play_audio"] = gr.outputs.Audio("filepath")
    
    # def start_processing():
    pipeline_result = pipeline(input_file.name, vad_type, "zhaosheng")
    save_png_path = f"./png/{pipeline_result['raw_file_path'].split('/')[-1].replace('.wav','.png')}"
    os.makedirs(os.path.dirname(save_png_path),exist_ok=True)
    get_spectrogram(pipeline_result['raw_file_path'],save_png_path)
    output["audio_length"] = pipeline_result["total_duration"]
    output["play_audio"] = pipeline_result["raw_file_path"]

    vad_urls = pipeline_result.get("vad_urls", [])
    vad_urls = sorted(vad_urls, key=lambda x: float(x.split("/")[-1].split("_")[0]))
    vad_players = create_audio_players(vad_urls)
    output["vad_result"] = generate_html_content(vad_players,"VAD结果")

    output["play_vad_files"] = pipeline_result.get("play_vad_files", [])

    selected_urls = pipeline_result.get("selected_urls", [])
    selected_urls = sorted(selected_urls, key=lambda x: float(x.split("/")[-1].split("_")[0]))
    selected_players = create_audio_players(selected_urls)
    output["selected_result"] = generate_html_content(selected_players,"选择的片段")
    
    output["play_mandarin_files"] = pipeline_result.get("play_mandarin_files", [])
    output["embeddings"] = pipeline_result.get("embeddings", [])
    output["selected_times"] = pipeline_result.get("selected_times", [])
    output["url"] = pipeline_result.get("url", [])
    output["asr_result"] = pipeline_result.get("asr_result", "")
    output["nlp_result"] = pipeline_result.get("nlp_result", "")
    
    return_data = [
        save_png_path,
        output["play_audio"],
        output["vad_result"],
        output["selected_result"],
        output["embeddings"],
        output["selected_times"],
        output["asr_result"],
        output["nlp_result"],

    ]

    return return_data

inputs = [
    gr.inputs.File(label="上传音频文件"),
    gr.inputs.Radio(["nn_vad", "energybase_vad"], label="选择VAD类型"),
]
outputs = [
    gr.outputs.Image(label="频谱图", type="filepath"),
    gr.outputs.Audio(label="试听音频", type="filepath"),
    gr.outputs.HTML(label="VAD结果"),
    gr.outputs.HTML(label="普通话过滤结果"),
    gr.outputs.Textbox(label="特征提取结果"),
    gr.outputs.Textbox(label="聚类结果"),
    gr.outputs.Textbox(label="ASR结果"),
    gr.outputs.Textbox(label="文本分类结果"),
]

title = "音频处理Demo"
description = "这是一个演示音频处理的在线Demo，可以逐步输出处理结果。"
examples = [
    ["path/to/audio.wav", "nn_vad"],
    ["path/to/audio.wav", "energybase_vad"],
]

gr.Interface(
    fn=process_audio,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    # examples=examples
).launch(share=True,server_name="0.0.0.0",server_port=7860)
