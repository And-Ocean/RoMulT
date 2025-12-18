import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

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

# TODO: 从其他地方copy来的缺失模态处理方法
# 六种缺失模态组合，等概率抽取
_MISSING_CHOICES: Tuple[Tuple[str, ...], ...] = (
    ("L",),
    ("A",),
    ("V",),
    ("L", "A"),
    ("L", "V"),
    ("A", "V"),
)


def _stack_batch(batch: Iterable[Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
    """
    将 TensorDataset 输出的样本列表堆叠成一个 batch 字典。
    返回的张量均为 CPU tensor，后续在训练循环中再搬到对应设备。
    """
    input_ids, visual, acoustic, input_mask, segment_ids, label_ids = zip(*batch)

    visual_tensor = torch.stack(visual, dim=0).float()
    acoustic_tensor = torch.stack(acoustic, dim=0).float()

    # 可用性 mask：1 表示存在该模态，0 表示缺失
    visual_mask = torch.ones(visual_tensor.shape[:2], dtype=torch.bool)
    acoustic_mask = torch.ones(acoustic_tensor.shape[:2], dtype=torch.bool)

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "visual": visual_tensor,
        "acoustic": acoustic_tensor,
        "attention_mask": torch.stack(input_mask, dim=0),
        "token_type_ids": torch.stack(segment_ids, dim=0),
        "labels": torch.stack(label_ids, dim=0).float(),
        "visual_mask": visual_mask,
        "acoustic_mask": acoustic_mask,
    }


def _apply_missing_modalities(
    batch: Dict[str, torch.Tensor], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    在给定 batch 上逐样本随机屏蔽模态，返回缺失模态域的 batch。

    缺失策略：
        - 六种组合 (L, A, V, LA, LV, AV) 均等概率；
        - 被屏蔽的模态直接置零；
        - 文本缺失时，input_ids / attention_mask / token_type_ids 全部置为 padding。
    """
    missing = {k: (v.clone() if k != "labels" else v) for k, v in batch.items()}
    batch_size = missing["labels"].size(0)

    for idx in range(batch_size):
        keep_modalities = set(random.choice(_MISSING_CHOICES))

        if "L" not in keep_modalities:
            missing["input_ids"][idx].fill_(pad_token_id)
            missing["attention_mask"][idx].zero_()
            missing["token_type_ids"][idx].zero_()

        if "A" not in keep_modalities:
            missing["acoustic"][idx].zero_()
            missing["acoustic_mask"][idx].zero_()

        if "V" not in keep_modalities:
            missing["visual"][idx].zero_()
            missing["visual_mask"][idx].zero_()

    return missing


def domain_collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    新版：同一批样本复制一份，直接在拷贝上随机屏蔽模态生成缺失域。
    """
    full_batch = _stack_batch(batch)
    missing_batch = _apply_missing_modalities(full_batch, pad_token_id=0)
    return full_batch, missing_batch
