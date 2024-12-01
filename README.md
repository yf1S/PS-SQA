# PS-SQA: Pitch-and-Spectrum-Aware Singing Quality Assessment With Bias Correction And Model Fusion

Author: Yu-Fei Shi (University of Science and Technology of China) Email: zkddsr2023@mail.ustc.edu.cn



## Dependencies:

 * Fairseq toolkit:  https://github.com/pytorch/fairseq  Make sure you can `import fairseq` in Python.
 * torch, numpy, scipy, torchaudio
 * I have exported my conda environment for this project to `environment.yml`
 * You also need to download a pretrained wav2vec2 model checkpoint.  These can be obtained here:  https://github.com/pytorch/fairseq/tree/main/examples/wav2vec. 
 * You also need to have a MOS dataset. 

### Data preparation

Awaiting Release


### Inference from pretrained model

We provide a pretrained PH-SSL-MOS, to download it and run inference, run:

`python predict_vmc.py.py --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints/his_v1 --datadir vmos2024/track2/ --outfile answer_main.txt`

You should see the following output:

```
[UTTERANCE] Test error= 0.375977
[UTTERANCE] Linear correlation coefficient= 0.619588
[UTTERANCE] Spearman rank correlation coefficient= 0.610374
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.445954
[SYSTEM] Test error= 0.056024
[SYSTEM] Linear correlation coefficient= 0.863070
[SYSTEM] Spearman rank correlation coefficient= 0.857703
[SYSTEM] Kendall Tau rank correlation coefficient= 0.680672
```


### How to train

First, make sure you already have the dataset and one pretrained fairseq base model (e.g., `fairseq/wav2vec_small.pt`).

To run PH_SSL_MOS using the dataset, run:

`python PH_SSL_MOS.py --datadir vmos2024/track2/ --fairseq_base_model fairseq/wav2vec_small.pt`

Once the training has finished, checkpoints can be found in the `checkpoints` directory.  The best one is the one with the highest number.  To run inference using this checkpoint, run:

`python predict_vmc.py.py --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints/his_v1 --datadir vmos2024/track2/ --outfile answer_main.txt`

Similarly, CP_SSL_MOS and S_SSL_MOS can be trained


