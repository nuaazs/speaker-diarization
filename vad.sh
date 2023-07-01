#!/bin/bash

# step 1 vad
# ROOT_PATH="/home/zhaosheng/VAF_UTILS/utils/examples/ex16_vad_cpp"
if [ $# -ne 5 ];then
    echo "Usage: $0 <input.wav> <output.vad.wav> <output.txt> <smooth_threshold> <min_duration>"
    exit 1
fi
rm -rf $3
rm -rf $4

# get file name from $1 and add with random number
FILE_NAME=$(basename $1)_$(date +%s%N).bin

# extract pcm data from wav file
# use ffmpeg merge 2 channel wav to 1 channel
ffmpeg -i $1 -ac 1 -ar 16000 $(basename $1)_1c.wav -y > /dev/null 2>&1

ffmpeg -i $(basename $1)_1c.wav -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y ${FILE_NAME} > /dev/null 2>&1

rm -rf $(basename $1)_1c.wav

# apply voice activity detection
echo "* Doing ->   bin/vad ${FILE_NAME} $2 $3 $4 $5 $6"
bin/vad ${FILE_NAME} $2 $3 $4 $5 $6
# echo "* Doing ->    rm -f ${FILE_NAME}"
# remove intermediate binary file
rm -f ${FILE_NAME}


# 用途： 通过ffmpeg和apply-vad实现vad
# 通过命令行参数接收 1. wav文件地址<input.vad> 2. 保存vad后的wav文件地址<output.vad.wav>
# 实现过程
# 1. 通过ffmpeg 生成bin ： ffmpeg -i <input.vad.wav> -f s16le -acodec pcm_s16le -ar 44100 -map_metadata -1 output.bin
# 2. bin传给apply-wav : apply-vad <output.bin> <output.vad.wav>
# 3. 删除中间变量bin文件

# step 2 embedding
python encode.py --folder_path $2 --bin_save_path test_data.bin --txt_save_path file_index.txt
# step 3 kmeans
bin/kmeans test_data.bin output.txt 17 192 2 3 300 0.1 cos

# 获取output.txt文件中的标签和对应索引
labels=($(cut -d ',' -f 2 output.txt))
indices=($(cut -d ',' -f 1 output.txt))

# 逐行读取file_index.txt文件中的路径和索引，并按标签进行分类复制
while IFS=',' read -r file_path index; do
    label=${labels[$index]}  # 获取对应索引的标签
    dir="./$label"  # 创建以标签名为名称的文件夹
    mkdir -p "$dir"
    cp "$file_path" "$dir"  # 复制文件到相应文件夹中
done <file_index.txt
