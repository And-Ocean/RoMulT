import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch
import random

# Keep default tensors on CPU; move to CUDA explicitly in training.
torch.set_default_dtype(torch.float32)
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            # Convert per-emotion logits to binary targets of shape (4,)
            Y = torch.argmax(Y, dim=-1).long()
        return X, Y, META        

MISSING_COMBOS = [
    ("L",),
    ("A",),
    ("V",),
    ("L", "A"),
    ("L", "V"),
    ("A", "V"),
]


def domain_collate_fn(batch):
    """
    Pair each full sample with a miss-domain copy where random modality combinations are zeroed.
    """
    sample_inds, texts, audios, visions, labels, metas = [], [], [], [], [], []
    for X, Y, META in batch:
        sample_ind, text, audio, vision = X
        sample_inds.append(sample_ind)
        texts.append(text)
        audios.append(audio)
        visions.append(vision)
        labels.append(Y)
        metas.append(META)

    sample_ind_full = torch.tensor(sample_inds, dtype=torch.long)
    text_full = torch.stack(texts, dim=0)
    audio_full = torch.stack(audios, dim=0)
    vision_full = torch.stack(visions, dim=0)
    Y_batch = torch.stack(labels, dim=0)
    META_batch = metas

    text_miss = text_full.clone()
    audio_miss = audio_full.clone()
    vision_miss = vision_full.clone()
    missing_mask = torch.zeros(len(batch), 3, dtype=torch.bool)  # L, A, V

    for i in range(len(batch)):
        combo = random.choice(MISSING_COMBOS)
        if "L" not in combo:
            text_miss[i].zero_()
            missing_mask[i, 0] = True
        if "A" not in combo:
            audio_miss[i].zero_()
            missing_mask[i, 1] = True
        if "V" not in combo:
            vision_miss[i].zero_()
            missing_mask[i, 2] = True

    batch_X_full = (sample_ind_full, text_full, audio_full, vision_full)
    batch_X_miss = (sample_ind_full, text_miss, audio_miss, vision_miss)
    return batch_X_full, batch_X_miss, Y_batch, META_batch, missing_mask
