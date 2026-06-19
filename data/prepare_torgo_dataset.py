import os
import glob
import librosa
import soundfile as sf
import re
from tqdm import tqdm

# ================= 配置区 =================
# 你的 TORGO 原始数据集里 F01 文件夹的路径 (请根据实际情况修改)
data_people = 'F04'
RAW_TORGO_F01_DIR = "./data/TORGO/F04" 
# 处理后输出的数据集路径
OUTPUT_DIR = "./data/torgo_f04_pinyin"  
# ==========================================

wav_out_dir = os.path.join(OUTPUT_DIR, "wavs")
csv_out_path = os.path.join(OUTPUT_DIR, "metadata.csv")

# 创建输出文件夹
os.makedirs(wav_out_dir, exist_ok=True)

# 查找 F01 下的所有 Session
sessions = glob.glob(os.path.join(RAW_TORGO_F01_DIR, "Session*"))
if not sessions:
    print(f"❌ 在 {RAW_TORGO_F01_DIR} 下没有找到 Session 文件夹，请检查路径！")
    exit()

processed_count = 0
skipped_count = 0

print(f"🚀 开始处理 F01 数据，输出至: {OUTPUT_DIR}")

with open(csv_out_path, 'w', encoding='utf-8') as f_csv:
    for session in sessions:
        session_name = os.path.basename(session)
        prompts_dir = os.path.join(session, 'prompts')
        
        if not os.path.exists(prompts_dir):
            continue

        # 遍历所有文本 prompt
        txt_files = glob.glob(os.path.join(prompts_dir, '*.txt'))
        
        for txt_file in tqdm(txt_files, desc=f"处理 {session_name}"):
            basename = os.path.basename(txt_file).replace('.txt', '')

            # 1. 读取并清洗文本
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # 清理 TORGO 中常见的特殊标记 (如 [breaths], [noise])
            text = re.sub(r'\[.*?\]', '', text)
            # 替换多余的空格和换行
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 如果清理后文本为空，跳过
            if not text:
                skipped_count += 1
                continue

            # 2. 寻找对应的音频 (优先使用音质更好的 headMic)
            wav_file = os.path.join(session, 'wav_headMic', f"{basename}.wav")
            if not os.path.exists(wav_file):
                wav_file = os.path.join(session, 'wav_arrayMic', f"{basename}.wav")
            
            # 如果连阵列麦克风的录音也没有，跳过
            if not os.path.exists(wav_file):
                skipped_count += 1
                continue

            # 3. 处理音频 (加载、重采样为 24kHz, 单声道)
            try:
                # sr=24000 指定采样率，mono=True 转为单声道
                y, sr = librosa.load(wav_file, sr=24000, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # 过滤太短或太长的音频（防止模型对齐崩溃或显存溢出）
                if duration < 0.3 or duration > 15.0:
                    skipped_count += 1
                    continue

                # 4. 写入目标文件夹
                # 为了防止不同 Session 中有重名文件，我们在新文件名中加入 session_name
                out_wav_name = f"{data_people}_{session_name}_{basename}.wav"
                out_wav_path = os.path.join(wav_out_dir, out_wav_name)
                
                # 保存为 16-bit PCM WAV
                sf.write(out_wav_path, y, sr, subtype='PCM_16')
                
                # 5. 写入 metadata.csv (格式: wavs/文件名.wav|文本内容)
                f_csv.write(f"{out_wav_name}|{text}\n")
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ 处理音频 {wav_file} 时出错: {e}")
                skipped_count += 1

print("\n🎉 处理完成！")
print(f"✅ 成功处理并保留的有效音频: {processed_count} 条")
print(f"⏭️ 因缺失、太短/太长或无文本而跳过的音频: {skipped_count} 条")
print(f"📄 元数据文件已生成: {csv_out_path}")