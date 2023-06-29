接口文档：

**接口功能**：获取主要声音的接口

**请求URL**：`http://localhost:5001/get_main_voice`

**请求方法**：POST

**请求参数**：
- file_mode：文件模式，可选值为`file`或`url`，表示上传文件或使用文件链接，默认为`file`
- spkid：说话人ID，默认为`no_spkid`
- wav_file：上传的音频文件，当`file_mode`为`file`时必传
- wav_url：音频文件的链接地址，当`file_mode`为`url`时必传

**响应参数**：
- length：主要声音的总时长，单位为秒
- url：合成的主要声音音频文件的链接地址
- times：每段声音的开始和结束时间，格式为[(start_time, end_time), ...]
- error：错误码，0表示成功，1表示失败
- error_message：失败时的错误信息

**示例代码**：
```python
import requests
import wget
import os

ROOT = "/home/zhaosheng/speaker-diarization/data/speaker_diarisation_test_data"
files = [os.path.join(ROOT,_file) for _file in os.listdir(ROOT) if _file.endswith(".wav")]

for wav_file_path in files:
    spkid = wav_file_path.split("/")[-1].split(".")[0].split("-")[-2]
    url = "http://localhost:5001/get_main_voice"
    payload = {
        "file_mode": "file",
        "spkid": spkid,
    }
    files=[
        ('wav_file',(wav_file_path.split("/")[-1],open(wav_file_path,'rb'),'application/octet-stream'))
    ]
    response = requests.post(url, data=payload, files=files)
    print(response.json())
    url = response.json()["url"]
    wget.download(url, out=f"output/{spkid}.wav")
```

使用说明：

1. 将待处理的音频文件放置在指定的文件夹下，例如`/home/zhaosheng/speaker-diarization/data/speaker_diarisation_test_data`。
2. 修改示例代码中的`ROOT`变量为音频文件所在的文件夹路径。
3. 运行示例代码，它会遍历文件夹中的每个音频文件，并通过接口调用获取主要声音。
4. 获取到主要声音后，将其保存到指定的输出文件夹中，例如`output`文件夹。
5. 根据实际需求进行修改和定制。