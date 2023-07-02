import torchaudio
from speechbrain.pretrained import EncoderClassifier
# import glob
import torch
import cfg
random_seed = torch.randint(1000, 9999, (1,)).item()
language_id = EncoderClassifier.from_hparams(source="/VAF/src/nn/LANG", savedir=f"./pretrained_models/lang-id-ecapa", run_opts={"device":cfg.DEVICE})
language_id.eval()

def mandarin_filter(filelist,score_threshold=0.7):
    """
    Filter the mandarin audio
    """
    # read the wav file
    # waveform = wavdata.to(cfg.DEVICE)
    pass_list = []
    data_list = []
    for filepath in filelist:
        wavdata = torchaudio.load(filepath)[0].reshape(-1)
        data_list.append(wavdata)
    print(data_list[0].shape)
    print(len(data_list))
    # change data_list to tensor, padding to the same length

    wavdata = torch.nn.utils.rnn.pad_sequence(data_list,batch_first=True).to(cfg.DEVICE)
    result = language_id.classify_batch(wavdata)
    for _index in range(len(filelist)):
        score = result[1][_index].exp()
        if score > score_threshold and result[3][0].startswith("zh"):
            pass_list.append(filelist[_index])
        else:
            print(f"file: {filelist[_index]} is not mandarin, result:{result[1][_index]} {result[2][_index]}")
    return pass_list

# if __name__ == '__main__':
    # mandarin_filter()