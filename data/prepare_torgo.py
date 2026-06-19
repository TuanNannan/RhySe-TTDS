import os
import csv
import re
from pathlib import Path

def extract_torgo_to_csv(torgo_root_dir, output_file):
    torgo_path = Path(torgo_root_dir)
    dataset_entries = []
    skipped_count = 0

    if not torgo_path.exists():
        print(f"❌ 错误: 找不到数据集路径 {torgo_root_dir}")
        return

    # 正则规则 1: 匹配指令性方括号内容
    bracket_pattern = re.compile(r'\[.*?\]')
    # 正则规则 2: 匹配所有非字母、非空格的符号（剔除句号、逗号、问号等）
    punctuation_pattern = re.compile(r'[^\w\s]')

    print("正在进行深度清洗并导出 CSV...")

    for speaker_dir in torgo_path.iterdir():
        if not speaker_dir.is_dir() or not speaker_dir.name.startswith(('F', 'M')):
            continue

        for session_dir in speaker_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith('Session'):
                continue

            prompts_dir = session_dir / 'prompts'
            if not prompts_dir.exists():
                continue

            for txt_file in prompts_dir.glob('*.txt'):
                stem = txt_file.stem

                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        raw_text = f.read().strip()
                except UnicodeDecodeError:
                    with open(txt_file, 'r', encoding='latin-1') as f:
                        raw_text = f.read().strip()

                if not raw_text:
                    continue

                # --- 清洗逻辑 ---
                # 1. 剔除带方括号的指令数据
                if bracket_pattern.search(raw_text):
                    skipped_count += 1
                    continue
                
                # 2. 剔除无法辨认的数据
                if 'xxx' in raw_text.lower():
                    skipped_count += 1
                    continue

                # 3. 文本正则化：剔除标点符号并转为大写 (ASR 常规操作)
                # 先转大写，再删标点，最后合并多余空格
                clean_text = raw_text.upper()
                clean_text = punctuation_pattern.sub('', clean_text)
                clean_text = ' '.join(clean_text.split())

                # 如果清洗后文本变空了（虽然概率低），则跳过
                if not clean_text:
                    continue

                for mic_type in ['wav_headMic', 'wav_arrayMic']:
                    wav_file = session_dir / mic_type / f"{stem}.wav"
                    if wav_file.exists():
                        dataset_entries.append({
                            'audio_file': str(wav_file.absolute()),
                            'text': clean_text
                        })

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_file', 'text'], delimiter='|')
        writer.writeheader()
        writer.writerows(dataset_entries)

    print(f"✅ 深度清洗完成！")
    print(f"   - 成功保留: {len(dataset_entries)} 条音频-文本对")
    print(f"   - 剔除无效数据: {skipped_count} 条")
    print(f"   - 文本示例: {dataset_entries[0]['text'] if dataset_entries else 'N/A'}")

if __name__ == "__main__":
    TORGO_ROOT = "./TORGO" 
    OUTPUT_CSV = "torgo_manifest_final.csv"
    extract_torgo_to_csv(TORGO_ROOT, OUTPUT_CSV)