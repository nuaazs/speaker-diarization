
import gradio as gr
import os

# def get_tabs_of_wavfiles(input_files):
#     tabs = []
#     for file_path in input_files:
#         tab = gr.Audio(source=file_path,label="试听音频", type="filepath")
#         tabs.append(tab)
#     return tabs


def get_wav_files(folder):
    wav_files = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".wav"):
            wav_files.append(os.path.join(folder, file_name))
    return wav_files


def create_audio_players(audio_files):
    audio_players = []
    for _index,audio_file in enumerate(audio_files,1):
        
        # audio file is local file path on server

        audio_player = f"<audio controls><source src='{audio_file}' type='audio/wav'> </audio>"
        # show index of audio file in html
        audio_player += f"<p>第{_index}个片段，{audio_file}</p>"
        audio_players.append(audio_player)
    return audio_players


def generate_html_content(audio_players):
    html_content = "<div>"
    for audio_player in audio_players:
        html_content += audio_player
    html_content += "</div>"
    return html_content

def pipeline(filepath, vad_type, spkid):
    print([filepath] * 10)
    return {
        "raw_file_path": filepath,
        "text": "abc",
        "file_list": [filepath] * 10
    }

def process_audio(input_file, vad_type):
    output = pipeline(input_file.name, vad_type, "zhaosheng")
    # multi_wav 
    # tabs = get_tabs_of_wavfiles(output["file_list"])
    return_data = [
        # output["raw_file_path"],
        # output["text"],
        '<span class="mermaid">\ngraph LR\nBox1["Hi"] --> BoxHaha["Haha"]\n</span>\n<script src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>'
    ]
    return return_data

inputs = [
    gr.inputs.File(label="上传音频文件"),
    gr.inputs.Radio(["nn_vad", "energybase_vad"], label="选择VAD类型"),

    
]
outputs = [
    # gr.outputs.Audio(label="试听音频", type="filepath"),
    # gr.outputs.Textbox(label="ASR结果"),
    # preview several wav files
    gr.outputs.HTML()
    # gr.outputs.Tab(label="音频子片段预览")
]

title = "音频处理Demo"
description = "这是一个演示音频处理的在线Demo，可以逐步输出处理结果。"

gr.Interface(
    fn=process_audio,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    layout="vertical",

).launch(share=True,server_name="0.0.0.0",server_port=7861)
