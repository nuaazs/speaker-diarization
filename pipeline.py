from utils.vad import nn_vad, energybase_vad
from utils.encoder import encode_folder
from utils.cluster import find_optimal_subset
from utils.mandarin import mandarin_filter
from asr import file_upload_offline as asr_offline
from utils.nlp import classify_text
import subprocess

def pipeline(filepath,spkid):
    # return example
    # {'selected_times': [('51.81', '61.79'), ('7.14', '28.57')], 'url': 'http://192.168.3.169:9000/raw/zhaosheng_selected.wav', 'text': {'text': '你好，请问您这边是什么咨询林女士吗？嗯，好好，李女士，您好，我这里在淘宝商城客服中心的，今天给您致电，主要是来通知您，你们淘宝商城申请的八八VIP业务审核已经通过了，那包括移金中心会在您名下的账户扣款经理的名费八百八十八元。我这么通知您清楚吗？啊啊，你说您那边北记的吧，我这边说的吗？什么是你什么问题吗？申请了我们淘宝商城的八八VIP会员保对呀。然后呢嗯这现在是已经审核通过了吗？那包括营业中心会在您名下的账户扣款经理的名费，八百八十八元的里算清楚吗？不知道没有啊，我没有申请啊啊啊，您是说您是没有医打女士没有啊，没事，那我原哪一年不是只有八十八元，不是，那么您的查期值未满意期的话，扣的是八百八十八元的里啊，那您的查税值有满意金的吗？'}}
    tmp_folder = f"/tmp/{spkid}"
    # 步骤一： VAD
    wav_files = nn_vad(filepath,tmp_folder)
    print(f"\t * -> VAD结果: {wav_files}")
    # 步骤二： 普通话过滤
    wav_files = mandarin_filter(wav_files)
    print(f"\t * -> 普通话过滤结果: {wav_files}")
    # 步骤三： 提取特征
    file_emb = encode_folder(wav_files)
    # print(f"\t * -> 提取特征结果: {file_emb}")
    # 步骤四： 聚类
    selected_files,total_duration, url,wav_file_path,selected_times = find_optimal_subset(file_emb, spkid=spkid,similarity_threshold=0.8, save_wav_path=tmp_folder)
    print(f"\t * -> 聚类结果: {selected_times}")
    print(f"\t * -> 聚类结果URL: {url}")

    # 步骤五： ASR
    text = asr_offline(filepath)
    print(f"\t * -> ASR结果: {text}")
    # 步骤六： NLP
    nlp_result = classify_text(text)
    print(f"\t * -> 文本分类结果: {nlp_result}")
    
    subprocess.call(f"rm -rf {tmp_folder}",shell=True)
    return {
        "selected_times": selected_times,
        "url": url,
        "text": text,
        "text_tpye": nlp_result,
        "total_duration": total_duration
    }
if __name__ == "__main__":
    r = pipeline("/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav","zhaosheng")
    print(r)
