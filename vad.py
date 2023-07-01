import subprocess
def vad(filepath,save_folder_path,smooth_threshold=0.5,min_duration=2):
    bin_path = f"${filepath[:-4]}.bin"
    cmd = f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  {bin_path}> /dev/null 2>&1"
    subprocess.call(cmd,shell=True)
    vad_cmd = f"vad {bin_path} {save_folder_path} {save_folder_path}/result.txt {smooth_threshold} {min_duration}"
    subprocess.call(vad_cmd,shell=True)
    return