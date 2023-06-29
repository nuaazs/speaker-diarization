# coding = utf-8
# @Time    : 2022-09-05  15:04:36
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: models.

from speechbrain.pretrained import SpeakerRecognition
import torch
import cfg


emb = SpeakerRecognition.from_hparams(
    source="./ECAPATDNN",
    savedir=f"./pretrained_models/ECAPATDNN",
    run_opts={"device": cfg.DEVICE},
)