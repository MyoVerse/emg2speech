import torch
from torch import nn
import torch.nn.functional as F



class featuresNorm(nn.Module):
    """
    Expect input: (T, N, C, C')
    Applies BatchNorm2d over (C', T) for each channel C, independently per sample N.
    """
    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.bn2d = nn.BatchNorm2d(channels, eps = eps, momentum = momentum, affine = affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"featuresNorm expects (T,N,C,C), got {x.shape}"
        T, N, C, _ = x.shape
        assert C == self.channels, f"BatchNorm2d expects C={self.channels}, got {C}"

        x = x.movedim(0, -1)          
        x = self.bn2d(x)             
        x = x.movedim(-1, 0).contiguous()  
        return x


class RotationInvariantMLP(nn.Module):
   
    def __init__(self, inFeatures: int, mlpFeatures, pooling: str = "mean", offsets = (-1, 0, 1)) -> None:
        super().__init__()
        assert len(mlpFeatures) > 0
        layers = []
        fin = inFeatures
        for fout in mlpFeatures:
            layers += [nn.Linear(fin, fout), nn.ReLU()]
            fin = fout
        self.mlp = nn.Sequential(*layers)
        assert pooling in {"max", "mean"}
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"RotationInvariantMLP expects (T,N,C,C), got {x.shape}"
        x = torch.stack([x.roll(off, dims = 2) for off in self.offsets], dim = 2)   
        x = self.mlp(x.flatten(start_dim = 3))                                    
        return x.max(dim = 2).values if self.pooling == "max" else x.mean(dim = 2)  



class TDSConv2dBlock(nn.Module):
    def __init__(self, channels: int, width: int, kernelWidth: int, padMode: str = "replicate") -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.kernelWidth = kernelWidth
        self.padMode = padMode

        self.conv2d = nn.Conv2d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = (1, kernelWidth),
            padding = 0
        )
        self.relu = nn.ReLU()
        self.layerNorm = nn.LayerNorm(channels * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Tin, N, Cflat = x.shape
        assert Cflat == self.channels * self.width

        y = x.movedim(0, -1).reshape(N, self.channels, self.width, Tin)  
        padLeft = self.kernelWidth - 1

        if self.padMode == "zeros":
            y = F.pad(y, (padLeft, 0, 0, 0)) 
        elif self.padMode == "replicate":
            first = y[..., :1]                               
            pad = first.expand(-1, -1, -1, padLeft)        
            y = torch.cat([pad, y], dim = -1)
        elif self.padMode == "reflect":
           
            use = min(padLeft, Tin - 1)
            reflected = y[..., 1:use + 1].flip(-1)
            if use < padLeft:
                extra = y[..., :1].expand(-1, -1, -1, padLeft - use)
                pad = torch.cat([reflected, extra], dim = -1)
            else:
                pad = reflected
            y = torch.cat([pad, y], dim = -1)
        else:
            raise ValueError(f"Unknown padMode: {self.padMode}")

        y = self.conv2d(y)                               
        y = self.relu(y)
        y = y.reshape(N, Cflat, Tin).movedim(-1, 0)       
        y = y + x
        return self.layerNorm(y)



class TDSFullyConnectedBlock(nn.Module):
    
    def __init__(self, numFeatures: int) -> None:
        super().__init__()
        self.fcBlock = nn.Sequential(
            nn.Linear(numFeatures, numFeatures),
            nn.ReLU(),
            nn.Linear(numFeatures, numFeatures)
        )
        self.layerNorm = nn.LayerNorm(numFeatures)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fcBlock(x) + x
        return self.layerNorm(y)


class TDSConvEncoder(nn.Module):
    
    def __init__(self, numFeatures: int, blockChannels, kernelWidth: int) -> None:
        super().__init__()
        blocks = []
        for ch in blockChannels:
            assert numFeatures % ch == 0, f"num_features={numFeatures} must be divisible by channels={ch}"
            blocks.append(TDSConv2dBlock(ch, numFeatures // ch, kernelWidth))
            blocks.append(TDSFullyConnectedBlock(numFeatures))
        self.tdsBlocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tdsBlocks(x)


class DualHeadTDSCTC(nn.Module):
    """
    Input : (N, T, C, C)   
    Internally permute -> (T, N, C, C).
    Outputs:
      - unitLogprobs  : (T, N, U + 1)
      - phoneLogprobs : (T, N, P + 1)
    """
    def __init__(
        self,
        *,
        inFeatures: int,               
        mlpFeatures,                   
        blockChannels,                
        kernelWidth: int,              
        numUnits: int = 101,                 
        numPhones: int = 41,  
        unitBlank: int = 100,
        phoneBlank: int = 40,              
        electrodeChannels: int = 31,  
        bottleneckDim = 128
    ) -> None:
        super().__init__()
        self.C = electrodeChannels
        self.inFeatures = inFeatures
        self.unitBlank = unitBlank
        self.phoneBlank = phoneBlank

        self.featNorm = featuresNorm(channels = self.C)       
        self.riMlp    = RotationInvariantMLP(               
            inFeatures = inFeatures,
            mlpFeatures = mlpFeatures,
            pooling = "mean",
            offsets = (-1, 0, 1),
        )
        H = mlpFeatures[-1]
        self.encoder   = TDSConvEncoder(                     
            numFeatures = H,
            blockChannels = blockChannels,
            kernelWidth = kernelWidth,
        )

        self.post = nn.Sequential(
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, bottleneckDim),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.unitHead  = nn.Linear(bottleneckDim, numUnits)
        self.phoneHead = nn.Linear(bottleneckDim, numPhones)

        

    def forward(self, inputs: torch.Tensor):
        
        assert inputs.ndim == 4 and inputs.shape[2] == self.C and inputs.shape[3] == self.C, \
            f"Expected (N,T,{self.C},{self.C}), got {inputs.shape}"

        x = inputs.permute(1, 0, 2, 3).contiguous()    
        y = self.featNorm(x)                          
        y = self.riMlp(y)                            
        y = self.encoder(y)
        z = self.post(y)                           

        unitLogprobs  = F.log_softmax(self.unitHead(z),  dim = -1)  
        phoneLogprobs = F.log_softmax(self.phoneHead(z), dim = -1)  

        return {"unitLogprobs": unitLogprobs, "phoneLogprobs": phoneLogprobs}
