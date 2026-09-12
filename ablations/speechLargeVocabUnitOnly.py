"""
Ablation: GENERAL-VOCAB EMG-to-audio conversion with a UNIT HEAD ONLY.

The single training objective is the HuBERT-unit CTC loss.

For description of the data, please see largeVocabDataVisualization.ipynb.

Run:
    python ablations/speechLargeVocabUnitOnly.py
"""

from __future__ import annotations

import os
import sys
import pickle
from typing import List, Optional

import numpy as np
import Levenshtein
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import convModule
from emgJitter import computeConvSeq


"""
Paths and hyperparameters.
"""

DATA_PATH  = "/mnt/dataDrive/emg2Audio/cleanData/DATA.pkl"
UNITS_PATH = "/mnt/dataDrive/emg2Audio/cleanData/HuBERTLABELS.pkl"
TEXT_PATH  = "/mnt/dataDrive/emg2Audio/cleanData/textLABELS.pkl"

CKPT_DIR  = os.path.join(REPO, "ckpts", "unitOnlyConv")
CURVE_PDF = os.path.join(REPO, "ablations", "lossCurvesUnitOnly.pdf")
WAV_DIR   = os.path.join(REPO, "ablations", "unitOnlyWavs")

C = 31
fs = 5000
winMs, hopMs = 25.0, 20.0
shrinkAlpha = 0.1
DIAG = False
diagOnly = False

numUnits  = 101
unitBlank = 100

trainEnd, valEnd = 8500, 9260
batchSize = 32
numWorkers = 4
numberEpochs = 50
warmup = 5

dev = "cuda:0"

trainJitter = False

"""
Set > 0 to vocode that many of the best-decoded test utterances to wav
(requires textlesslib; imported lazily).
"""
numSynthExamples = 0


def zNormalize(DATA):
    """
    z-normalize the data along the time dimension.
    """
    normDATA = []
    for i in range(len(DATA)):
        Mean = np.mean(DATA[i], axis = -1)
        Std = np.std(DATA[i], axis = -1)
        normDATA.append((DATA[i] - Mean[..., np.newaxis])/Std[..., np.newaxis])
    return normDATA


class UnitJitterEMGDataset(Dataset):
    """
    emgJitter.EpochJitterEMGDataset with the phone targets dropped.

    Returns (covSeq, unitSeq, unitLen)
      covSeq  : (F, C, C) float32  — SPD feature sequence
      unitSeq : (Lu,)     int64    — HuBERT unit ids (0..99, 100 = blank)
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
        shrinkAlpha: float = 0.1,
        diag: bool = False,
        diagOnly: bool = False,
        eigenvectors: Optional[torch.Tensor] = None,
        jitter: bool = True,
    ):
        nItems = len(inputsList)
        assert nItems == len(unitLabelsList) == len(unitLabelLengths), \
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
        self.shrinkAlpha = float(shrinkAlpha)
        self.diag = bool(diag)
        self.diagOnly = bool(diagOnly)

        self.E = None
        if eigenvectors is not None:
            self.E = torch.as_tensor(eigenvectors, dtype = torch.float32, device = "cpu").contiguous()
            assert self.E.shape == (self.C, self.C), f"eigenvectors must be ({self.C},{self.C})"

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
            win = int(round(self.winMs * self.fs / 1000.0))
            if T < win:
                self.phases.append(0)
                continue
            maxPhase = min(hop - 1, T - win)
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

        covSeq, nFrames = computeConvSeq(
            X,
            fs = self.fs,
            winMs = self.winMs,
            hopMs = self.hopMs,
            phase = phase,
            shrinkAlpha = self.shrinkAlpha,
            diag = self.diag,
            diagOnly = self.diagOnly,
            eigenvectors = self.E
        )

        unitSeq = torch.as_tensor(self.unitLabels[idx], dtype = torch.long)
        unitLen = int(self.unitLabelLengths[idx])

        if unitSeq.numel() < unitLen:
            raise ValueError(f"Unit target shorter than unitLen: {unitSeq.numel()} < {unitLen}")

        return covSeq.contiguous(), unitSeq, unitLen


@torch.no_grad()
def padCollateUnit(batch, C: int, padValue: int = -1):
    """
    batch: list of tuples:
      (covSeq, unitSeq, unitLen)
    Returns:
      inputs: (N, Tmax, C, C)
      unitTargets : (N, LuMax) [padded with padValue]
      inputLengths : (N,)
      unitTargetLengths : (N,)
    """
    covSeqs, unitSeqs, unitLens = zip(*batch)
    B = len(batch)

    Tmax = max(int(x.shape[0]) for x in covSeqs)
    inputs = covSeqs[0].new_zeros((B, Tmax, C, C))
    I = torch.eye(C, dtype = inputs.dtype)
    for b, x in enumerate(covSeqs):
        Tb = int(x.shape[0])
        inputs[b, :Tb] = x
        if Tb < Tmax:
            inputs[b, Tb:Tmax] = I

    LuMax = max(int(L) for L in unitLens)
    unitTargets = torch.full((B, LuMax), padValue, dtype = torch.long)
    for b, (u, Lu) in enumerate(zip(unitSeqs, unitLens)):
        u = torch.as_tensor(u).view(-1)
        unitTargets[b, :int(Lu)] = u[:int(Lu)]

    inputLengths      = torch.as_tensor([int(t.shape[0]) for t in covSeqs], dtype = torch.int32)
    unitTargetLengths = torch.as_tensor(unitLens, dtype = torch.int32)

    return inputs, unitTargets, inputLengths, unitTargetLengths


class UnitHeadTDSCTC(nn.Module):
    """
    convModule.DualHeadTDSCTC with the phone head (and its linear layer) and the
    output-side block removed. The unit head reads the encoder output directly.

    Input : (N, T, C, C)
    Internally permute -> (T, N, C, C).
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

        assert inputs.ndim == 4 and inputs.shape[2] == self.C and inputs.shape[3] == self.C, \
            f"Expected (N,T,{self.C},{self.C}), got {inputs.shape}"

        x = inputs.permute(1, 0, 2, 3).contiguous()
        y = self.featNorm(x)
        y = self.riMlp(y)
        y = self.encoder(y)

        unitLogprobs = F.log_softmax(self.unitHead(y), dim = -1)

        return {"unitLogprobs": unitLogprobs}


def _concatTargets(padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    assert padded.ndim == 2 and lengths.ndim == 1
    return torch.cat([padded[b, :int(L)] for b, L in enumerate(lengths.tolist())],
                     dim = 0).to(dtype = torch.long, copy = False)


def trainOperation(model, device, trainLoader, optimizer, ctcUnit):
    model.train()
    total = 0.0

    for (inputs, unitTgts, inLens, unitLens) in trainLoader:
        inputs   = inputs.to(device, non_blocking = True)
        unitTgts = unitTgts.to(device, non_blocking = True)
        inLens   = inLens.to(device, dtype = torch.long, non_blocking = True)
        unitLens = unitLens.to(device, dtype = torch.long, non_blocking = True)

        unitTargets1d = _concatTargets(unitTgts, unitLens)

        optimizer.zero_grad(set_to_none = True)

        out   = model(inputs)
        logpU = out['unitLogprobs']

        loss = ctcUnit(logpU, unitTargets1d, inLens, unitLens)
        loss.backward()
        optimizer.step()

        total += float(loss.item())

    return total / max(1, len(trainLoader))


@torch.no_grad()
def valOperation(model, device, valLoader, ctcUnit):
    model.eval()
    total = 0.0

    for (inputs, unitTgts, inLens, unitLens) in valLoader:
        inputs   = inputs.to(device, non_blocking = True)
        unitTgts = unitTgts.to(device, non_blocking = True)
        inLens   = inLens.to(device, dtype = torch.long, non_blocking = True)
        unitLens = unitLens.to(device, dtype = torch.long, non_blocking = True)

        unitTargets1d = _concatTargets(unitTgts, unitLens)

        out   = model(inputs)
        logpU = out['unitLogprobs']

        total += float(ctcUnit(logpU, unitTargets1d, inLens, unitLens).item())

    return total / max(1, len(valLoader))


@torch.no_grad()
def testOperation(model, device, testLoader, ctcUnit):
    model.eval()
    outputsList = []

    lossUtotal = 0.0
    numBatches = 0

    for (inputs, unitTgts, inLens, unitLens) in testLoader:
        inputs   = inputs.to(device)
        unitTgts = unitTgts.to(device)
        inLens   = inLens.to(device)
        unitLens = unitLens.to(device)

        unitTargets1d = _concatTargets(unitTgts, unitLens)

        out   = model(inputs)
        logpU = out["unitLogprobs"]

        batchLossU = ctcUnit(logpU, unitTargets1d, inLens, unitLens)

        outputsList.append({"unitLogprobs": logpU.transpose(0, 1).detach().cpu()})

        lossUtotal += float(batchLossU.item())
        numBatches += 1

    metrics = {"lossUnit": lossUtotal / max(1, numBatches)}
    return outputsList, metrics


"""
Simple beam-search algorithm.
"""

def ctcPrefixBeamSearch(
    logProbs,
    testLen = None,
    beamSize = 5,
    blank = unitBlank,
    topk = None,
    allowDoubles = True,
):

    lp = np.asarray(logProbs)
    Ttotal, V = lp.shape
    T = Ttotal if testLen is None else int(min(testLen, Ttotal))

    beams = {(): (0.0, -np.inf)}

    def add(store, seq, addPb, addPnb):
        if seq in store:
            pb, pnb = store[seq]
            if addPb  != -np.inf: pb  = np.logaddexp(pb,  addPb)
            if addPnb != -np.inf: pnb = np.logaddexp(pnb, addPnb)
            store[seq] = (pb, pnb)
        else:
            store[seq] = (addPb, addPnb)

    for t in range(T):
        row = lp[t]
        new = {}

        if topk is not None and topk < V:
            cand = np.argpartition(row, -topk)[-topk:]
            if blank not in cand:
                worstIdx = cand[np.argmin(row[cand])]
                cand[cand == worstIdx] = blank
        else:
            cand = range(V)

        for seq, (pb, pnb) in beams.items():
            add(new, seq, np.logaddexp(pb, pnb) + row[blank], -np.inf)

            last = seq[-1] if seq else None

            for c in cand:
                if c == blank:
                    continue
                pC = row[c]

                if c == last:

                    add(new, seq, -np.inf, pnb + pC)

                    if allowDoubles:
                        add(new, seq + (c,), -np.inf, pb + pC)
                else:
                    add(new, seq + (c,), -np.inf, np.logaddexp(pb, pnb) + pC)

        if len(new) > beamSize:
            items = sorted(new.items(),
                           key = lambda kv: np.logaddexp(*kv[1]),
                           reverse = True)[:beamSize]
            beams = dict(items)
        else:
            beams = new

    bestSeq = max(beams.items(), key = lambda kv: np.logaddexp(*kv[1]))[0]
    return bestSeq


def findClosestTranscription(decodedTranscript, unitTranscription):

    dist = Levenshtein.distance(decodedTranscript, unitTranscription)

    return dist


def plotLossCurves(tLoss, vLoss, outPath = CURVE_PDF):
    t = np.array(tLoss, dtype = float)
    v = np.array(vLoss, dtype = float)
    epochs = np.arange(1, t.shape[0] + 1)
    name = r'$\mathcal{L}_{\mathrm{CTC}}^{\mathrm{unit}}$'

    plt.figure()
    trainLine, = plt.plot(epochs, t)
    valLine,   = plt.plot(epochs, v, linestyle = ":", alpha = 0.8,
                          color = trainLine.get_color())

    plt.xlabel("Epoch", fontsize = 20)
    plt.ylabel("Loss", fontsize = 20)
    plt.legend(
        [trainLine, valLine],
        [f"Train {name}", f"Val {name}"],
        loc = "upper right",
        frameon = True,
        columnspacing = 1.5,
        handlelength = 2.5,
        fontsize = 15
    )
    plt.tight_layout()
    plt.savefig(outPath, bbox_inches = "tight")
    plt.close()


def synthesizeExamples(decodedOut, levs, unitLENGTHS, textLABELS, device, howMany):
    """
    Vocode the best-decoded test utterances back to audio, as in the last cell of
    speechLargeVocab.ipynb (writes wavs instead of an inline Audio widget).
    """
    import soundfile as sf
    sys.path.insert(0, "/home/k2/src/textlesslib")
    from textless.vocoders.tacotron2.vocoder import TacotronVocoder

    vocoder = TacotronVocoder.by_name("hubert-base-ls960", "kmeans", 100).to(device)
    os.makedirs(WAV_DIR, exist_ok = True)

    indices = np.argsort(np.array(levs)/np.array(unitLENGTHS))
    for which in indices[:howMany]:
        which = int(which)
        units = torch.tensor(decodedOut[which], dtype = torch.long, device = device)
        wav = vocoder(units)
        sf.write(os.path.join(WAV_DIR, f"{which}.wav"), wav.cpu().numpy(), vocoder.output_sample_rate)
        print(which, levs[which]/unitLENGTHS[which], textLABELS[valEnd + which])


def main():
    os.makedirs(CKPT_DIR, exist_ok = True)
    device = torch.device(dev if torch.cuda.is_available() else "cpu")

    DATA       = pickle.load(open(DATA_PATH, "rb"))
    unitLABELS = np.load(UNITS_PATH, allow_pickle = True)
    textLABELS = pickle.load(open(TEXT_PATH, "rb"))

    """
    Pad the HuBERT unit sequences to a common length (to be used with CTC loss).
    """
    MAX = max(len(u) for u in unitLABELS)
    unitizedLabels = np.zeros((len(unitLABELS), MAX), dtype = np.int64)
    for i, seq in enumerate(unitLABELS):
        unitizedLabels[i, :len(seq)] = np.asarray(seq, dtype = np.int64)
    unitLabelLengths = np.array([len(seq) for seq in unitLABELS], dtype = np.int32)

    normDATA = zNormalize(DATA)

    emgTrain, emgVal, emgTest = normDATA[:trainEnd], normDATA[trainEnd:valEnd], normDATA[valEnd:]
    huTrain, huVal, huTest    = unitizedLabels[:trainEnd], unitizedLabels[trainEnd:valEnd], unitizedLabels[valEnd:]
    huLTrain, huLVal, huLTest = unitLabelLengths[:trainEnd], unitLabelLengths[trainEnd:valEnd], unitLabelLengths[valEnd:]

    trainDS = UnitJitterEMGDataset(
        emgTrain, huTrain, huLTrain,
        fs = fs, winMs = winMs, hopMs = hopMs,
        shrinkAlpha = shrinkAlpha, diag = DIAG, diagOnly = diagOnly,
        jitter = trainJitter,
    )
    valDS = UnitJitterEMGDataset(
        emgVal, huVal, huLVal,
        fs = fs, winMs = winMs, hopMs = hopMs,
        shrinkAlpha = shrinkAlpha, diag = DIAG, diagOnly = diagOnly,
        jitter = False,
    )
    testDS = UnitJitterEMGDataset(
        emgTest, huTest, huLTest,
        fs = fs, winMs = winMs, hopMs = hopMs,
        shrinkAlpha = shrinkAlpha, diag = DIAG, diagOnly = diagOnly,
        jitter = False,
    )

    trainLoader = DataLoader(
        trainDS, batch_size = batchSize, shuffle = True,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateUnit(b, C = C),
    )
    valLoader = DataLoader(
        valDS, batch_size = batchSize, shuffle = False,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateUnit(b, C = C),
    )
    testLoader = DataLoader(
        testDS, batch_size = 1, shuffle = False,
        num_workers = numWorkers, pin_memory = True,
        collate_fn = lambda b: padCollateUnit(b, C = C),
    )

    model = UnitHeadTDSCTC(
        inFeatures        = C * C,
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

    plotLossCurves(tLoss, vLoss)
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
        logp = outputs[i]["unitLogprobs"]
        logpNp = logp.squeeze(0).numpy()
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

    if numSynthExamples > 0:
        synthesizeExamples(decodedOut, levs, unitLENGTHS, textLABELS, device, numSynthExamples)


if __name__ == "__main__":
    main()
