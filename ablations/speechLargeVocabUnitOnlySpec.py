"""
Ablation: GENERAL-VOCAB EMG-to-audio conversion from EMG SPECTROGRAMS, vec(B).

This is the vec(B) row of table 1 in the paper.

Run:
    python ablations/speechLargeVocabUnitOnlySpec.py
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convModule
from speechLargeVocabUnitOnly import (
    zNormalize,
    _concatTargets,
    trainOperation,
    valOperation,
    testOperation,
    ctcPrefixBeamSearch,
    findClosestTranscription,
    plotLossCurves,
)


"""
Paths and hyperparameters.
"""

DATA_PATH  = "/mnt/dataDrive/emg2Audio/cleanData/DATA.pkl"
UNITS_PATH = "/mnt/dataDrive/emg2Audio/cleanData/HuBERTLABELS.pkl"

CKPT_DIR  = os.path.join(REPO, "ckpts", "unitOnlySpec")
CURVE_PDF = os.path.join(REPO, "ablations", "lossCurvesUnitOnlySpec.pdf")

C = 31
fs = 5000
winMs, hopMs = 25.0, 20.0
targetBins = 31
nFft = 256
useLog = True

numUnits  = 101
unitBlank = 100

trainEnd, valEnd = 8500, 9260
batchSize = 32
numWorkers = 4
numberEpochs = 50
warmup = 5

trainJitter = False

dev = "cuda:0"


@torch.no_grad()
def computeSpecSeq(
    X: torch.Tensor,
    *,
    fs: int,
    winMs: float,
    hopMs: float,
    phase: int = 0,
    targetBins: int = 31,
    nFft: int = 256,
    useLog: bool = True,
    eps: float = 1e-12,
    bandLow: float = 80.0,
    bandHigh: float = 1000.0,
) -> Tuple[torch.Tensor, int]:
    """
    Band-limited power spectrogram, pooled to targetBins bands, in dB.

    Returns:
      specSeq: (F, C, B) where B = targetBins
      nFrames: int
    """
    C, T = X.shape
    if T <= 0:
        out = torch.full((1, C, targetBins), -12.0, dtype = torch.float32)
        return out, 1

    win = int(round(winMs * fs / 1000.0))
    hop = int(round(hopMs * fs / 1000.0))
    if win <= 0 or hop <= 0:
        raise ValueError("winMs and hopMs must produce positive sample sizes")
    if nFft < win:
        nFft = int(1 << (max(win - 1, 0)).bit_length())

    maxPhase = max(0, T - 1)
    phase = min(max(phase, 0), maxPhase)
    Xs = X[:, phase:]
    Tp = Xs.shape[1]

    if Tp < win:
        Xs = F.pad(Xs, (0, win - Tp))

    window = torch.hann_window(win, dtype = Xs.dtype, device = Xs.device)
    S = torch.stft(
        Xs, n_fft = nFft, hop_length = hop, win_length = win,
        window = window, center = False, return_complex = True
    )

    power = S.abs().pow(2)

    nyq = fs * 0.5
    bandHighEff = min(bandHigh, nyq)
    freqs = torch.fft.rfftfreq(nFft, d = 1.0/fs).to(Xs.device)
    bandMask = (freqs >= bandLow) & (freqs <= bandHighEff)
    if not torch.any(bandMask):
        raise ValueError(
            f"No FFT bins fall in [{bandLow}, {bandHighEff}] Hz for nFft={nFft}, fs={fs}"
        )

    bandPower = power[:, bandMask, :]
    FBand = bandPower.shape[1]
    if FBand < targetBins:
        targetBins = FBand

    pooled = F.adaptive_avg_pool1d(bandPower.transpose(1, 2), output_size = targetBins)
    pooled = pooled * (FBand / float(targetBins))
    pooled = pooled.transpose(1, 2)

    if useLog:
        pooled = pooled.clamp_min(eps)
        pooled = 10.0 * torch.log10(pooled)

    frames = pooled.shape[-1]
    specSeq = pooled.permute(2, 0, 1).contiguous().to(dtype = torch.float32)
    return specSeq, frames


class SpecJitterEMGDataset(Dataset):
    """
    emgJitterSpec.EpochJitterDataset with the unit targets named as in this repo.

    Returns (specSeq, unitSeq, unitLen)
      specSeq : (F, C, B) float32
      unitSeq : (Lu,)     int64
      unitLen : int
    """
    def __init__(
        self,
        inputsList: List[torch.Tensor] | List,
        unitLabelsList: List,
        unitLabelLengths: List[int],
        *,
        fs: int = 5000,
        winMs: float = 25.0,
        hopMs: float = 20.0,
        targetBins: int = 31,
        nFft: int = 256,
        useLog: bool = True,
        jitter: bool = True,
    ):
        assert len(inputsList) == len(unitLabelsList) == len(unitLabelLengths), \
               "Inputs/labels/lengths must have the same length."

        self.inputs: List[torch.Tensor] = []
        for X in inputsList:
            Xt = torch.as_tensor(X, dtype = torch.float32, device = "cpu")
            assert Xt.ndim == 2, "Each EMG sample must be (C, T)"
            self.inputs.append(Xt.contiguous())

        self.unitLabels       = unitLabelsList
        self.unitLabelLengths = unitLabelLengths

        self.C = int(self.inputs[0].shape[0])
        self.fs = int(fs)
        self.winMs = float(winMs)
        self.hopMs = float(hopMs)
        self.targetBins = int(targetBins)
        self.nFft = int(nFft)
        self.useLog = bool(useLog)

        self.jitterEnabled = bool(jitter)
        self.phases: List[int] = []
        self.resampleJitter(seed = None)

    def resampleJitter(self, seed: Optional[int] = None) -> None:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        self.phases = []

        hop = int(round(self.hopMs * self.fs / 1000.0))
        hop = max(hop, 1)

        for X in self.inputs:
            T = int(X.shape[1])
            if T < 1:
                self.phases.append(0)
                continue
            maxPhase = min(hop - 1, max(0, T - 1))
            if self.jitterEnabled and maxPhase > 0:
                p = int(torch.randint(0, maxPhase + 1, (1,), generator = g))
            else:
                p = 0
            self.phases.append(p)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        X = self.inputs[idx]
        phase = self.phases[idx]

        specSeq, nFrames = computeSpecSeq(
            X,
            fs = self.fs,
            winMs = self.winMs,
            hopMs = self.hopMs,
            phase = phase,
            targetBins = self.targetBins,
            nFft = self.nFft,
            useLog = self.useLog,
        )

        unitSeq = torch.as_tensor(self.unitLabels[idx], dtype = torch.long)
        unitLen = int(self.unitLabelLengths[idx])

        if unitSeq.numel() < unitLen:
            raise ValueError(f"Unit target shorter than unitLen: {unitSeq.numel()} < {unitLen}")

        return specSeq.contiguous(), unitSeq, unitLen


@torch.no_grad()
def padCollateSpec(batch, C: int, padValue: int = -1):
    """
    batch: list of tuples:
      (specSeq, unitSeq, unitLen)
    Returns:
      inputs: (N, Tmax, C, B)
      unitTargets : (N, LuMax) [padded with padValue]
      inputLengths : (N,)
      unitTargetLengths : (N,)

    Frames past each sample's length are filled with the identity, as in
    book1Spec.ipynb; CTC ignores them through inputLengths.
    """
    specSeqs, unitSeqs, unitLens = zip(*batch)
    B = len(batch)

    Tmax = max(int(x.shape[0]) for x in specSeqs)
    inputs = specSeqs[0].new_zeros((B, Tmax, C, C))
    I = torch.eye(C, dtype = inputs.dtype)
    for b, x in enumerate(specSeqs):
        Tb = int(x.shape[0])
        inputs[b, :Tb] = x
        if Tb < Tmax:
            inputs[b, Tb:Tmax] = I

    LuMax = max(int(L) for L in unitLens)
    unitTargets = torch.full((B, LuMax), padValue, dtype = torch.long)
    for b, (u, Lu) in enumerate(zip(unitSeqs, unitLens)):
        u = torch.as_tensor(u).view(-1)
        unitTargets[b, :int(Lu)] = u[:int(Lu)]

    inputLengths      = torch.as_tensor([int(t.shape[0]) for t in specSeqs], dtype = torch.int32)
    unitTargetLengths = torch.as_tensor(unitLens, dtype = torch.int32)

    return inputs, unitTargets, inputLengths, unitTargetLengths


class SpecUnitHeadTDSCTC(nn.Module):
    """
    codes7/convModule.TDSConvCTCModule, written to return the dict this repo's
    train/val/test operations expect.

    Input : (N, T, C, B)
    Output:
      - unitLogprobs : (T, N, U + 1)
    """
    def __init__(
        self,
        *,
        inFeatures: int,
        mlpFeatures,
        blockChannels,
        kernelWidth: int,
        numUnits: int = 101,
        unitBlank: int = 100,
        electrodeChannels: int = 31
    ) -> None:
        super().__init__()
        self.C = electrodeChannels
        self.inFeatures = inFeatures
        self.unitBlank = unitBlank

        self.featNorm = convModule.featuresNorm(channels = self.C)
        self.riMlp    = convModule.RotationInvariantMLP(
            inFeatures = inFeatures,
            mlpFeatures = mlpFeatures,
            pooling = "mean",
            offsets = (-1, 0, 1),
        )
        H = mlpFeatures[-1]
        self.encoder  = convModule.TDSConvEncoder(
            numFeatures = H,
            blockChannels = blockChannels,
            kernelWidth = kernelWidth,
        )
        self.unitHead = nn.Linear(H, numUnits)

    def forward(self, inputs: torch.Tensor):

        assert inputs.ndim == 4 and inputs.shape[2] == self.C, \
            f"Expected (N,T,{self.C},B), got {inputs.shape}"

        x = inputs.permute(1, 0, 2, 3).contiguous()
        y = self.featNorm(x)
        y = self.riMlp(y)
        y = self.encoder(y)

        unitLogprobs = F.log_softmax(self.unitHead(y), dim = -1)

        return {"unitLogprobs": unitLogprobs}


def main():
    os.makedirs(CKPT_DIR, exist_ok = True)
    device = torch.device(dev if torch.cuda.is_available() else "cpu")

    DATA       = pickle.load(open(DATA_PATH, "rb"))
    unitLABELS = np.load(UNITS_PATH, allow_pickle = True)

    MAX = max(len(u) for u in unitLABELS)
    unitizedLabels = np.zeros((len(unitLABELS), MAX), dtype = np.int64)
    for i, seq in enumerate(unitLABELS):
        unitizedLabels[i, :len(seq)] = np.asarray(seq, dtype = np.int64)
    unitLabelLengths = np.array([len(seq) for seq in unitLABELS], dtype = np.int32)

    normDATA = zNormalize(DATA)

    emgTrain, emgVal, emgTest = normDATA[:trainEnd], normDATA[trainEnd:valEnd], normDATA[valEnd:]
    huTrain, huVal, huTest    = unitizedLabels[:trainEnd], unitizedLabels[trainEnd:valEnd], unitizedLabels[valEnd:]
    huLTrain, huLVal, huLTest = unitLabelLengths[:trainEnd], unitLabelLengths[trainEnd:valEnd], unitLabelLengths[valEnd:]

    dsArgs = dict(fs = fs, winMs = winMs, hopMs = hopMs, targetBins = targetBins,
                  nFft = nFft, useLog = useLog)
    trainDS = SpecJitterEMGDataset(emgTrain, huTrain, huLTrain, jitter = trainJitter, **dsArgs)
    valDS   = SpecJitterEMGDataset(emgVal,   huVal,   huLVal,   jitter = False, **dsArgs)
    testDS  = SpecJitterEMGDataset(emgTest,  huTest,  huLTest,  jitter = False, **dsArgs)

    trainLoader = DataLoader(
        trainDS, batch_size = batchSize, shuffle = True,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateSpec(b, C = C),
    )
    valLoader = DataLoader(
        valDS, batch_size = batchSize, shuffle = False,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateSpec(b, C = C),
    )
    testLoader = DataLoader(
        testDS, batch_size = 1, shuffle = False,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateSpec(b, C = C),
    )

    model = SpecUnitHeadTDSCTC(
        inFeatures        = C * targetBins,
        mlpFeatures       = [384],
        blockChannels     = [24, 24, 24, 24],
        kernelWidth       = 14,
        numUnits          = numUnits,
        unitBlank         = unitBlank,
        electrodeChannels = C
    ).to(device)

    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number trainable params: {numParams:,}")

    ctcUnit = nn.CTCLoss(blank = unitBlank, zero_infinity = True)

    Optimizer = optim.AdamW(
        model.parameters(), lr = 3e-4, weight_decay = 1e-4, betas = (0.9, 0.98)
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        Optimizer,
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(Optimizer, start_factor = 0.1, total_iters = warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max = numberEpochs - warmup, eta_min = 1e-6),
        ],
        milestones = [warmup],
    )

    valLOSS, tLoss, vLoss = [], [], []
    for epoch in range(1, numberEpochs + 1):

        trainDS.resampleJitter(seed = None)

        trainLoss = trainOperation(model, device, trainLoader, Optimizer, ctcUnit)
        valLoss   = valOperation(model, device, valLoader, ctcUnit)

        sched.step()

        valLOSS.append(valLoss)
        tLoss.append(trainLoss)
        vLoss.append(valLoss)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{epoch}.pt"))

        print(f"Epoch {epoch}/{numberEpochs}  Train {trainLoss:.4f}  Val {valLoss:.4f}\n")

    plotLossCurves(tLoss, vLoss, CURVE_PDF)
    np.save(os.path.join(CKPT_DIR, "valLoss.npy"), valLOSS)

    valLoss = np.load(os.path.join(CKPT_DIR, "valLoss.npy"))
    print(np.min(valLoss))
    print(np.argmin(valLoss) + 1)
    epoch = int(np.argmin(valLoss) + 1)

    modelWeight = torch.load(os.path.join(CKPT_DIR, f"{epoch}.pt"), weights_only = True)
    model.load_state_dict(modelWeight)

    outputs, metrics = testOperation(model, device, testLoader, ctcUnit)
    print(metrics)

    decodedOut = []
    for i in range(len(outputs)):
        logpNp = outputs[i]["unitLogprobs"].squeeze(0).numpy()
        decodedOut.append(
            ctcPrefixBeamSearch(logProbs = logpNp, beamSize = 1, blank = unitBlank)
        )

    levs = []
    unitLENGTHS = []
    for i in range(len(decodedOut)):
        unitLen = int(unitLabelLengths[valEnd + i])
        unitLENGTHS.append(unitLen)
        levs.append(findClosestTranscription(decodedOut[i], unitizedLabels[valEnd + i][:unitLen]))

    print(np.array(levs)/np.array(unitLENGTHS))
    print("Mean length of sentences: ", np.mean(unitLENGTHS))
    print("Mean unit errors (insertion errors + deletion errors + substitution errors): ", np.mean(levs))
    print("Percent unit error: ", np.sum(levs)/np.sum(unitLENGTHS))


if __name__ == "__main__":
    main()
