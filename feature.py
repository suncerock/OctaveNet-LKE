import torch
import torch.nn as nn
from nnAudio import features

from data import INDEX_TO_KEY

class CQTWithShift(nn.Module):
    def __init__(self, sr=22050, hop_length=0.2, bins_per_octave=24, num_octaves=6) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.bins_per_semitone = bins_per_octave // 12
        self.num_octaves = num_octaves

        hop_frames = int(sr * hop_length)
        self.cqt_layer = features.cqt.CQT(
            sr=sr, hop_length=hop_frames, n_bins=(num_octaves + 1)*bins_per_octave, bins_per_octave=bins_per_octave)

    def forward(self, x, y, shift=True):

        with torch.no_grad():
            s = self.cqt_layer(x)
            s = torch.log(s + 1e-7)

            shift = torch.randint(low=-4, high=7+1, size=(len(x), 1)) if shift else torch.zeros((len(x), 1), dtype=torch.int64)

            s = s[torch.arange(len(s))[:, None], self.bins_per_semitone * (4 + shift) + torch.arange(self.num_octaves * self.bins_per_octave)]
            s = (s - torch.mean(s, dim=(1, 2), keepdim=True)) / torch.std(s, dim=(1, 2), keepdim=True)

            shift = shift.to(x.device)
            #TODO: Check whether this is unnecessary
            if len(y.shape) == 3:
                shift = shift.unsqueeze(dim=-1)
            y = torch.where(y != -1, (y - shift * 2) % len(INDEX_TO_KEY), y)

            s = s.unsqueeze(dim=1)
        return s, y
