import cfg
from utils.cmd import run_cmd

def vad_wav(input_data_path, save_folder,txt_save_path):
    cmd = f"{cfg.VAD_BIN} {input_data_path} {save_folder} {txt_save_path} 0.5 2"
    # print(cmd)
    run_cmd(cmd)
    return
