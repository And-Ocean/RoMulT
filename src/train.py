import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

from src.DA_Loss import CMD_Loss, SWD_Loss

KEEP_COMBOS = [
    ("L",),
    ("A",),
    ("V",),
    ("L", "A"),
    ("L", "V"),
    ("A", "V"),
]


def apply_random_mask(text, audio, vision, per_sample=True, return_mask=False):
    """
    Apply a random missing-modality mask to the given batch tensors.
    """
    t_m, a_m, v_m = text.clone(), audio.clone(), vision.clone()
    batch_size = t_m.size(0)
    mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=text.device)

    if per_sample:
        for i in range(batch_size):
            combo = random.choice(KEEP_COMBOS)
            if "L" not in combo:
                t_m[i].zero_()
                mask[i, 0] = True
            if "A" not in combo:
                a_m[i].zero_()
                mask[i, 1] = True
            if "V" not in combo:
                v_m[i].zero_()
                mask[i, 2] = True
    else:
        combo = random.choice(KEEP_COMBOS)
        if "L" not in combo:
            t_m.zero_()
            mask[:, 0] = True
        if "A" not in combo:
            a_m.zero_()
            mask[:, 1] = True
        if "V" not in combo:
            v_m.zero_()
            mask[:, 2] = True

    if return_mask:
        return t_m, a_m, v_m, mask
    return t_m, a_m, v_m


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    device = hyp_params.device if hyp_params.use_cuda else torch.device("cpu")
    if hyp_params.use_cuda:
        model = model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    # Use explicit CrossEntropy for IEMOCAP's 4 binary heads; fall back to configured loss otherwise.
    criterion = nn.CrossEntropyLoss() if hyp_params.dataset == 'iemocap' else getattr(nn, hyp_params.criterion)()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    writer = SummaryWriter(log_dir=os.path.join("runs", hyp_params.name))
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,
                'writer': writer,
                'device': device}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    

    scheduler = settings['scheduler']
    writer = settings['writer']
    device = settings['device']

    device_idx = device.index if hyp_params.use_cuda and device.type == "cuda" else None
    device_ids = [device_idx] if device_idx is not None else None

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        da_weight = getattr(hyp_params, 'da_weight', 0.1)
        miss_weight = getattr(hyp_params, 'miss_weight', 1.0)
        # da_loss_fn = CMD_Loss(n_moments=getattr(hyp_params, 'cmd_k', 5))
        da_loss_fn = SWD_Loss(num_projections = 128, p=2)
        if hyp_params.use_cuda:
            da_loss_fn = da_loss_fn.to(device)
        epoch_cls_sum = 0.0
        epoch_cmd_sum = 0.0
        for i_batch, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)):
            batch_X, batch_Y, batch_META = batch
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)

            model.zero_grad()

            if hyp_params.use_cuda:
                text = text.to(device)
                audio = audio.to(device)
                vision = vision.to(device)
                eval_attr = eval_attr.to(device)
                if hyp_params.dataset == 'iemocap':
                    eval_attr = eval_attr.long()

            batch_size = text.size(0)
            use_dp = hyp_params.use_cuda and batch_size > 10 and torch.cuda.device_count() > 1
            net = nn.DataParallel(model, device_ids=device_ids, output_device=device_idx) if use_dp else model

            do_robust = (da_weight > 0 or miss_weight > 0) and batch_size >= 2
            step_cmd = 0.0

            if do_robust:
                half = batch_size // 2
                text_full, audio_full, vision_full = text[:half], audio[:half], vision[:half]
                target_full = eval_attr[:half]

                # Paired missing view from the same samples
                text_miss_raw, audio_miss_raw, vision_miss_raw = text_full.clone(), audio_full.clone(), vision_full.clone()
                target_miss = target_full
                text_miss, audio_miss, vision_miss, miss_mask = apply_random_mask(
                    text_miss_raw, audio_miss_raw, vision_miss_raw, per_sample=True, return_mask=True
                )

                text_in = torch.cat([text_full, text_miss], dim=0)
                audio_in = torch.cat([audio_full, audio_miss], dim=0)
                vision_in = torch.cat([vision_full, vision_miss], dim=0)

                preds, features = net(text_in, audio_in, vision_in)
                preds_full, preds_miss = preds[:half], preds[half:]
                feat_full, feat_miss = features[:half], features[half:]
            else:
                preds_full, feat_full = net(text, audio, vision)
                preds_miss = feat_miss = None
                target_full = eval_attr

            if hyp_params.dataset == 'iemocap':
                reshaped_preds_full = preds_full.view(-1, 2)
                target_full_flat = target_full.view(-1)
                cls_loss = criterion(reshaped_preds_full, target_full_flat)
                step_cls = cls_loss.item()

                total_loss = cls_loss
                if do_robust:
                    reshaped_preds_miss = preds_miss.view(-1, 2)
                    target_miss_flat = target_miss.view(-1)
                    cls_loss_miss = criterion(reshaped_preds_miss, target_miss_flat)
                    cmd_loss = torch.tensor(0.0, device=device)
                    if da_weight > 0:
                        cmd_loss = da_loss_fn(feat_full.detach(), feat_miss)
                    total_loss = total_loss + miss_weight * cls_loss_miss + da_weight * cmd_loss
                    step_cls += miss_weight * cls_loss_miss.item()
                    if da_weight > 0:
                        step_cmd = cmd_loss.item()
            else:
                total_loss = criterion(preds_full, target_full)
                step_cls = total_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += total_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += total_loss.item() * batch_size
            epoch_cls_sum += step_cls * batch_size
            epoch_cmd_sum += step_cmd * batch_size

        denom = max(hyp_params.n_train, 1)
        writer.add_scalar("train/cls_loss", epoch_cls_sum / denom, epoch)
        writer.add_scalar("train/cmd_loss", epoch_cmd_sum / denom, epoch)
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False, mask_type=None):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        total_f1 = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    text = text.to(device)
                    audio = audio.to(device)
                    vision = vision.to(device)
                    eval_attr = eval_attr.to(device)
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                use_dp = hyp_params.use_cuda and batch_size > 10 and torch.cuda.device_count() > 1
                net = nn.DataParallel(model, device_ids=device_ids, output_device=device_idx) if use_dp else model
                if mask_type == "random":
                    text, audio, vision = apply_random_mask(text, audio, vision, per_sample=True, return_mask=False)
                preds, _ = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)              # (B*4, 2)
                    eval_attr = eval_attr.view(-1)         # (B*4,)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds.cpu())
                truths.append(eval_attr.cpu())
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        if hyp_params.dataset == 'iemocap':
            preds_label = torch.argmax(results, dim=-1)
            total_f1 = f1_score(truths.numpy(), preds_label.numpy(), average="weighted")
            if getattr(hyp_params, "diagnostics", False):
                eval_iemocap_diagnostics(results, truths, prefix="test/" if test else "val/")
        return avg_loss, results, truths, total_f1

    best_score = float("-inf") if (getattr(hyp_params, 'da_weight', 0) > 0 or getattr(hyp_params, 'miss_weight', 0) > 0) else float("inf")
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss_full, _, _, val_f1_full = evaluate(model, criterion, test=False, mask_type=None)
        val_loss_miss = val_f1_miss = None
        if getattr(hyp_params, 'da_weight', 0) > 0 or getattr(hyp_params, 'miss_weight', 0) > 0:
            _, _, _, val_f1_miss = evaluate(model, criterion, test=False, mask_type="random")
        test_loss, _, _, _ = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss_full)    # Decay learning rate by validation loss

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss_full, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)
        if val_f1_full is not None:
            writer.add_scalar("val/f1_full", val_f1_full, epoch)
        if val_f1_miss is not None:
            writer.add_scalar("val/f1_miss", val_f1_miss, epoch)

        print(f"Epoch {epoch:02d} | Time {duration:5.2f}s | Train Loss {train_loss:5.4f} | Valid Loss {val_loss_full:5.4f} | Test Loss {test_loss:5.4f}")
        if val_f1_full is not None:
            print(f"  Val F1 (full): {val_f1_full:.4f}")
        if val_f1_miss is not None:
            print(f"  Val F1 (random mask): {val_f1_miss:.4f}")

        robust_mode = getattr(hyp_params, 'da_weight', 0) > 0 or getattr(hyp_params, 'miss_weight', 0) > 0
        if robust_mode:
            val_score = (val_f1_full + (val_f1_miss if val_f1_miss is not None else 0)) / (2 if val_f1_miss is not None else 1)
            improved = val_score > best_score
        else:
            val_score = val_loss_full
            improved = val_score < best_score

        if improved:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt! (val_score={val_score:.4f})")
            save_model(hyp_params, model, name=hyp_params.name)
            best_score = val_score

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, _ = evaluate(model, criterion, test=True)
    writer.close()

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
