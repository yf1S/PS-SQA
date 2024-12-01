
import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import librosa
import warnings
random.seed(1984)

class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.dense1 = nn.Linear(self.ssl_features, 128)
        self.layernorm = nn.LayerNorm(248)
        self.dense2 = nn.Sequential(
            nn.Linear(248, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        
    def forward(self, wav, pitchs):
        wav = wav.squeeze(1)  
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.dense1(x)
        x_concat_pitch = torch.cat([x, pitchs], dim = -1).float()
        x_concat_pitch = self.layernorm(x_concat_pitch)
        x_concat_pitch = self.dense2(x_concat_pitch)
        
        return x_concat_pitch.squeeze(1)

    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0] + '.wav'
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        wav2, _ = librosa.load(wavpath,sr=16000)

        pitch, t = pw.dio(
            wav2.astype(np.float64),
            16000,
            frame_period=320 /16000 * 1000,
        )
        index_array = []
        pitch = pw.stonemask(wav2.astype(np.float64), pitch, t, 16000)
        index_values = [10 * (12 * np.log2(p / 440) % 12) for p in pitch if p != 0]
        index_array.append(index_values)
        count_array = np.zeros((len(index_array), 120), dtype=int)
        pitch_histogram = []
        for i, row in enumerate(index_array):
            length = len(row)
            for value in row:
                if value >= 120 or value < 0:
                    raise ValueError
                else:
                    count_array[i, int(np.floor(value))] += 1
        normalized_row = count_array[i] / length
        pitch_histogram.append(normalized_row)
        return wav, score, wavname, pitch_histogram
    

    def __len__(self):
        return len(self.wavnames)
    
    def remove_silence(self, y, window, hop, threshold):
        frames = librosa.util.frame(y, window, hop)
        sil_frames = []
        for i, f in enumerate(frames.transpose()):
            if np.sqrt(np.mean(f * f)) < threshold:
                sil_frames.append(i)
        y = np.delete(frames, sil_frames, 1)
        y = y.reshape(-1)
        return y

    def collate_fn(self, batch):  
        wavs, scores, wavnames, pitch_histograms= zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]  
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        pitch_histograms = torch.tensor(pitch_histograms).squeeze(1)
        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames, pitch_histograms

def systemID(uttID):
    return uttID.split('-')[2]
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='vmos2024/track2/checkpoint', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir 
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    SSL_OUT_DIM = 768

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)


    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_TEST_SYS_SRCC=0
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames, pitch_histograms = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            pitch_histograms = pitch_histograms.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, pitch_histograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        predictions = { }  # 字典，内容为filename : prediction
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, labels, filenames, pitch_histograms = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            pitch_histograms = pitch_histograms.to(device)
            outputs = net(inputs, pitch_histograms)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            output = outputs.cpu().detach().numpy()[0]
            predictions[filenames[0]] = output 
            
        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))

        true_MOS = { }
        testf = open(validlist, 'r')
        for line in testf:
            parts = line.strip().split(',')
            uttID = parts[0] + '.wav'
            MOS = float(parts[1])
            true_MOS[uttID] = MOS

        sorted_uttIDs = sorted(predictions.keys())
        #UTT_level
        ts = []
        ps = []
        for uttID in sorted_uttIDs:
            t = true_MOS[uttID]
            p = predictions[uttID]
            ts.append(t)
            ps.append(p)

        truths = np.array(ts)
        preds = np.array(ps)
        test_utt_SRCC=scipy.stats.spearmanr(truths.T, preds.T)[0]
    ### SYSTEM
        true_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop1 = true_sys_MOSes.setdefault(sysID, [ ])
            true_sys_MOSes[sysID].append(true_MOS[uttID])
        true_sys_MOS_avg = { }
        for k, v in true_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            true_sys_MOS_avg[k] = avg_MOS

        pred_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop = pred_sys_MOSes.setdefault(sysID, [ ])
            pred_sys_MOSes[sysID].append(predictions[uttID])
        pred_sys_MOS_avg = { }
        for k, v in pred_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
        pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
        sys_p = [ ]
        sys_t = [ ]
        for sysID in pred_sysIDs:
            sys_p.append(pred_sys_MOS_avg[sysID])
            sys_t.append(true_sys_MOS_avg[sysID])

        sys_true = np.array(sys_t)
        sys_predicted = np.array(sys_p)
        test_sys_SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)[0]

#save checkpoints
        print('[UTTERANCE] Spearman rank correlation coefficient= %f' % test_utt_SRCC)
        print('[system] Spearman rank correlation coefficient= %f' % test_sys_SRCC)
        if test_sys_SRCC > PREV_TEST_SYS_SRCC:
            PREV_TEST_SYS_SRCC=test_sys_SRCC
            print('sys_SRCC has decreased')
            path = os.path.join(ckptdir, 'ckpt_' + str(epoch))
            torch.save(net.state_dict(), path)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('SRCC has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break

    print('Finished Training')
