import argparse
import csv
import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.utils import get_data


def load_model_from_path(path, device):
    model = torch.load(path, map_location="cpu")
    if hasattr(model, "to"):
        model = model.to(device)
    model.eval()
    return model


def apply_modality_mask(text, audio, vision, combo):
    keep = set(combo)
    if "L" not in keep:
        text = text.clone()
        text.zero_()
    if "A" not in keep:
        audio = audio.clone()
        audio.zero_()
    if "V" not in keep:
        vision = vision.clone()
        vision.zero_()
    return text, audio, vision


@torch.no_grad()
def eval_one(model, loader, device, combo):
    preds_all = []
    labels_all = []

    for batch_X, batch_Y, _ in loader:
        _, text, audio, vision = batch_X
        labels = batch_Y.squeeze(-1).long()

        text = text.to(device)
        audio = audio.to(device)
        vision = vision.to(device)
        labels = labels.to(device)

        text_m, audio_m, vision_m = apply_modality_mask(text, audio, vision, combo)

        logits, _ = model(text_m, audio_m, vision_m)
        logits_2 = logits.view(-1, 2)
        labels_flat = labels.view(-1)

        preds = torch.argmax(logits_2, dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(labels_flat.cpu())

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Compare modalities on IEMOCAP for baseline vs DA models")
    parser.add_argument("--dataset", type=str, default="iemocap")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--aligned", action="store_true", default=True, help="use aligned data (default: True)")
    parser.add_argument("--unaligned", action="store_false", dest="aligned", help="use unaligned data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--baseline_ckpt", type=str, default="baseline")
    parser.add_argument("--da_ckpt", type=str, default="da")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--out_csv", type=str, default="RoMulT/csv", help="optional path to save csv results")
    parser.add_argument("--device", type=int, default=0, help="cuda device id to use (default: 0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    if use_cuda:
        torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")

    dataset = str.lower(args.dataset.strip())
    try:
        data = get_data(args, dataset, split=args.split)
    except FileNotFoundError as e:
        if not args.aligned:
            # Fallback to aligned cache if unaligned is missing.
            args.aligned = True
            data = get_data(args, dataset, split=args.split)
        else:
            raise e
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    baseline_path = os.path.join("/data/hhy/RoMulT/pre_trained_models", args.baseline_ckpt, ".pt")
    da_path = os.path.join("/data/hhy/RoMulT/pre_trained_models", args.da_ckpt, ".pt")

    baseline = load_model_from_path(baseline_path, device)
    da_model = load_model_from_path(da_path, device)

    combos = ["L", "A", "V", "LA", "LV", "AV", "LAV"]
    rows = []
    print("combo | base_acc | base_f1 | da_acc | da_f1 | acc_delta | f1_delta")
    for combo in combos:
        base_acc, base_f1 = eval_one(baseline, loader, device, combo)
        da_acc, da_f1 = eval_one(da_model, loader, device, combo)
        rows.append((combo, base_acc, base_f1, da_acc, da_f1, da_acc - base_acc, da_f1 - base_f1))
        print(f"{combo:4s} | {base_acc:.4f} | {base_f1:.4f} | {da_acc:.4f} | {da_f1:.4f} | {da_acc - base_acc:+.4f} | {da_f1 - base_f1:+.4f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["combo", "base_acc", "base_f1", "da_acc", "da_f1", "acc_delta", "f1_delta"])
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
