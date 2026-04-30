# `emg2speech`

If you use this dataset, code, or audio samples, please cite:

```bibtex
@inproceedings{emg2speech2026,
  title     = {emg2speech: synthesizing speech from electromyography using self-supervised speech models},
  author    = {Gowda, Harshavardhan T. and others},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).},
  year      = {2026}
}

Please see the notebooks for deails about model training. Download all data from the link below. 

```text
We present a neuromuscular speech interface that translates electromyographic (EMG) 
signals recorded from orofacial muscles during speech articulation directly into audio. 
We find that self-supervised speech representations (SS) are strongly linearly related 
to the electrical power of muscle activity: a simple linear mapping predicts EMG power 
from SS with a correlation of r = 0.85. In addition, EMG power vectors associated with 
distinct articulatory gestures form structured, separable clusters. Together, these 
observations suggest that SS implicitly encode articulatory mechanisms, as reflected 
in EMG activity. Leveraging this structure, we map EMG signals into the SS space and 
synthesize speech, enabling end-to-end EMG-to-speech generation without explicit 
articulatory modeling or vocoder training. We demonstrate this system with a participant 
with amyotrophic lateral sclerosis (ALS), converting orofacial EMG recorded while 
she silently articulated speech into audio.

1. Download the data from: https://osf.io/65vbx/ (under Files/Box).

(https://osf.io/65vbx/files/box - the OSF site shows them as 0B since they 
are hosted externally. 
You can still click on individual files and download them. 0B is misleading.)

2. 
└── tokens.txt # phoneme mapping.
├── GeneralCorpusData/
│   ├── DATA.pkl # General corpus EMG data from a healthy subject.
│   ├── textLABELS.pkl # text labels for general corpus data.
│   ├── HuBERTLABELS.pkl # HuBERT units from Google TTS audio of text labels (not synced to EMG).
|   ├── groundTruthAudioFiles.pkl # Ground truth subject recorded audio.
|   ├── synthesizedAudios/ # synthesized audio from EMG (all 400 sentences in the test set).
└── ALS_Data/
``` ├── DATA.pkl # EMG data from an ALS subject.
    ├── textLABELS.pkl # text labels for ALS corpus data.
    ├── HuBERTLABELS.pkl # HuBERT units from Google TTS audio of text labels (not synced to EMG).
    ├── dataSplit.npy # We randomly generated 60 indices to be used as test set.
    ├── synthesizedAudios/ # synthesized audio from EMG (all 60 sentences in the test set).   