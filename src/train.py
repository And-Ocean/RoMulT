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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

from DA_Loss import DA_Loss


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    device = hyp_params.device if hyp_params.use_cuda else torch.device("cpu")
    if hyp_params.use_cuda:
        model = model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    # Use explicit CrossEntropy for IEMOCAP's 4 binary heads; fall back to configured loss otherwise.
    criterion = nn.CrossEntropyLoss() if hyp_params.dataset == 'iemocap' else getattr(nn, hyp_params.criterion)()
    ctc_criterion = None
    ctc_a2l_module, ctc_v2l_module = None, None
    ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)
    writer = SummaryWriter(log_dir=os.path.join("runs", hyp_params.name))
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
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
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']
    writer = settings['writer']
    device = settings['device']
    

    device_idx = device.index if hyp_params.use_cuda and device.type == "cuda" else None
    device_ids = [device_idx] if device_idx is not None else None

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        da_weight = getattr(hyp_params, 'da_weight', 0.0)
        da_loss_fn = DA_Loss(n_moments=getattr(hyp_params, 'cmd_k', 5))
        if hyp_params.use_cuda:
            da_loss_fn = da_loss_fn.to(device)
        global_step_base = (epoch - 1) * len(train_loader)
        for i_batch, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)):
            if da_weight > 0:
                batch_X_full, batch_X_miss, batch_Y, batch_META, missing_mask = batch
                _, text_miss, audio_miss, vision_miss = batch_X_miss
            else:
                batch_X_full, batch_Y, batch_META = batch
                text_miss = audio_miss = vision_miss = None
                missing_mask = None

            sample_ind, text_full, audio_full, vision_full = batch_X_full
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                text_full = text_full.to(device)
                audio_full = audio_full.to(device)
                vision_full = vision_full.to(device)
                eval_attr = eval_attr.to(device)
                if da_weight > 0:
                    text_miss = text_miss.to(device)
                    audio_miss = audio_miss.to(device)
                    vision_miss = vision_miss.to(device)
                    if missing_mask is not None:
                        missing_mask = missing_mask.to(device)
                if hyp_params.dataset == 'iemocap':
                    eval_attr = eval_attr.long()
            
            batch_size = text_full.size(0)
            batch_chunk = hyp_params.batch_chunk
            
            ctc_loss = 0
                
            combined_loss = 0
            net = nn.DataParallel(model, device_ids=device_ids, output_device=device_idx) if batch_size > 10 and hyp_params.use_cuda else model
            step_cls = 0.0
            step_cmd = 0.0
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text_full.chunk(batch_chunk, dim=0)
                audio_chunks = audio_full.chunk(batch_chunk, dim=0)
                vision_chunks = vision_full.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                if da_weight > 0:
                    text_miss_chunks = text_miss.chunk(batch_chunk, dim=0)
                    audio_miss_chunks = audio_miss.chunk(batch_chunk, dim=0)
                    vision_miss_chunks = vision_miss.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, features_i = net(text_i, audio_i, vision_i)
                    if da_weight > 0:
                        preds_miss_i, features_miss_i = net(text_miss_chunks[i], audio_miss_chunks[i], vision_miss_chunks[i])

                    if hyp_params.dataset == 'iemocap':
                        reshaped_preds = preds_i.view(-1, 2)              # (B*4, 2)
                        reshaped_labels = eval_attr_i.view(-1)            # (B*4,)
                        cls_loss_i = criterion(reshaped_preds, reshaped_labels) / batch_chunk
                        if da_weight > 0:
                            cmd_loss_i = da_loss_fn(features_i, features_miss_i) / batch_chunk
                        else:
                            cmd_loss_i = 0.0
                        raw_loss_i = cls_loss_i + da_weight * cmd_loss_i
                        step_cls += cls_loss_i.item()
                        step_cmd += cmd_loss_i.item() if torch.is_tensor(cmd_loss_i) else float(cmd_loss_i)
                    else:
                        raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                        step_cls += raw_loss_i.item()

                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                ctc_loss.backward()
                combined_loss = raw_loss + ctc_loss
                if batch_chunk > 0:
                    step_cls /= batch_chunk
                    step_cmd /= batch_chunk
            else:
                preds, features = net(text_full, audio_full, vision_full)
                if da_weight > 0:
                    preds_miss, features_miss = net(text_miss, audio_miss, vision_miss)
                if hyp_params.dataset == 'iemocap':
                    reshaped_preds = preds.view(-1, 2)              # (B*4, 2)
                    reshaped_labels = eval_attr.view(-1)            # (B*4,)
                    cls_loss = criterion(reshaped_preds, reshaped_labels)
                    cmd_loss = da_loss_fn(features, features_miss) if da_weight > 0 else 0.0
                    raw_loss = cls_loss + da_weight * cmd_loss
                    step_cls = cls_loss.item()
                    step_cmd = cmd_loss.item() if torch.is_tensor(cmd_loss) else float(cmd_loss)
                else:
                    raw_loss = criterion(preds, eval_attr)
                    step_cls = raw_loss.item()
                combined_loss = raw_loss + ctc_loss
                combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            global_step = global_step_base + i_batch
            writer.add_scalar("train/cls_loss", step_cls, global_step)
            writer.add_scalar("train/cmd_loss", step_cmd, global_step)
            writer.add_scalar("train/raw_loss", raw_loss.item(), global_step)
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
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
                
                net = nn.DataParallel(model, device_ids=device_ids, output_device=device_idx) if batch_size > 10 and hyp_params.use_cuda else model
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
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion, None, None, None, None, None)
        val_loss, _, _ = evaluate(model, None, None, criterion, test=False)
        test_loss, _, _ = evaluate(model, None, None, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)

        print(f"Epoch {epoch:02d} | Time {duration:5.2f}s | Train Loss {train_loss:5.4f} | Valid Loss {val_loss:5.4f} | Test Loss {test_loss:5.4f}")
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)
    writer.close()

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
