import torch
from src.dataset import Multimodal_Datasets

# 配置可自行调整
DATA_PATH = 'data'
DATASET = 'iemocap'
SPLIT = 'train'
ALIGN = True
NUM_SAMPLES = 30

if __name__ == '__main__':
    ds = Multimodal_Datasets(DATA_PATH, data=DATASET, split_type=SPLIT, if_align=ALIGN)
    print(f"Dataset split={SPLIT}, aligned={ALIGN}")
    print(f"Num samples: {len(ds)}")
    print(f"Text shape: {ds.text.shape}")
    print(f"Audio shape: {ds.audio.shape}")
    print(f"Vision shape: {ds.vision.shape}")
    print(f"Labels shape: {ds.labels.shape}")
    print('-'*40)
    for idx in range(min(NUM_SAMPLES, len(ds))):
        X, Y, META = ds[idx]
        sample_ind, text, audio, vision = X
        print(f"Sample {idx}")
        print(f"  text shape: {text.shape}")
        print(f"  audio shape: {audio.shape}")
        print(f"  vision shape: {vision.shape}")
        print(f"  labels: {Y}")
        print(f"  META: {META}")
        print('-'*40)
