import gradio as gr
import os
import logging
import torch
import soundfile
import torchaudio

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)


os.environ["MODELSCOPE_CACHE"] = "./"
online_inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='/home/zhaosheng/asr_damo_websocket/online/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.6',
    mode="paraformer_streaming",
    device='gpu:3',
)
offline_inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='/home/zhaosheng/asr_damo_websocket/online/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    model_revision="v1.2.4",
    device='gpu:2',
)

inference_pipline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision="v1.1.7",device='gpu:2')



def transcribe_audio(audio):
    audio = audio.reshape(-1).numpy()
    sample_offset = 0
    chunk_size = [5, 10, 5] #[5, 10, 5] 600ms, [8, 8, 4] 480ms
    stride_size = chunk_size[1] * 960
    param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size}
    final_result = ""

    for sample_offset in range(0, len(audio), min(stride_size, len(audio) - sample_offset)):
        if sample_offset + stride_size >= len(audio) - 1:
            stride_size = len(audio) - sample_offset
            param_dict["is_final"] = True
        rec_result = online_inference_pipeline(audio_in=audio[sample_offset: sample_offset + stride_size],
                                        param_dict=param_dict)
        if len(rec_result) != 0:
            final_result += rec_result['text'] + " "
            print(rec_result)
    return final_result

def file_upload(file):
    audio, sample_rate = torchaudio.load(file.name)
    transcription = transcribe_audio(audio)
    return transcription

def transcribe_audio_offline(audio):
    audio = audio.reshape(-1).numpy()
    rec_result = offline_inference_pipeline(audio_in=audio)
    return rec_result['text']

def file_upload_offline(file):
    audio, sample_rate = torchaudio.load(file)
    transcription = transcribe_audio_offline(audio)
    transcription = inference_pipline(text_in=transcription)
    return transcription

# iface = gr.Interface(
#     fn=file_upload,
#     inputs=gr.inputs.File(label="Upload Audio File (Streaming ASR)"),
#     outputs="text",
#     title="LongYuan ASR",
#     theme="huggingface",
#     icon="🎤"
# )

# iface2 = gr.Interface(
#     fn=file_upload_offline,
#     inputs=gr.inputs.File(label="Upload Audio File (Non-streaming ASR)"),
#     outputs="text",
#     title="LongYuan ASR",
#     theme="huggingface",
#     icon="🎤"
# )

# # iface.launch(share=True,server_name="0.0.0.0",server_port=7860)
# iface2.launch(share=True,server_name="0.0.0.0",server_port=7860)

# # iface.launch(share=True,server_name="0.0.0.0",server_port=7860)