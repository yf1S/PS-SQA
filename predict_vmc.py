# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from PH_SSL_MOS import MosPredictor
from torch.utils.data.dataset import Dataset
import numpy as np
import torchaudio
import librosa
import pyworld as pw
import scipy.stats
import matplotlib.pyplot as plt

def systemID(uttID):
    return uttID.split('-')[2]

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = 'voicemos2024-track2-' + parts[0] 
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
    
    def collate_fn(self, batch):  ## zero padding
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    datadir = args.datadir
    outfile = os.path.join('', args.outfile)


    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

   
    SSL_OUT_DIM = 768
   
    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))
    wavdir = os.path.join(datadir, 'wav')
    validlist = "./test_mos_list.txt"

    print('Loading data')
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=1, collate_fn=validset.collate_fn)

    predictions = { }  
    print('Starting prediction')
    
    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames, pitch_histograms = data#wav=tensor(1,1,T) ,mos=tensor��1��,wavname=('sysc2f1e-utta38a09c.wav',)
        inputs = inputs.to(device)
        labels = labels.to(device)
        pitch_histograms = pitch_histograms.to(device)
        outputs = model(inputs,pitch_histograms)
        output = float(outputs.cpu().detach().numpy()[0])
        predictions[filenames[0]] = output  
           
    true_MOS = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = 'voicemos2024-track2-' + parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS
   
    
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    print('[UTTERANCE] Test error= %f' % MSE)
    LCC=np.corrcoef(truths, preds)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(truths, preds)
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

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

    MSE=np.mean((sys_true-sys_predicted)**2)
    print('[SYSTEM] Test error= %f' % MSE)
    LCC=np.corrcoef(sys_true, sys_predicted)
    print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)
    print('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    outfile = ''
    ans = open(outfile, 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()
if __name__ == '__main__':
    main()
