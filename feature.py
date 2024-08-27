import torch
import torch.nn as nn
from nnAudio import features

from data import INDEX_TO_KEY


class CQTWithShift(nn.Module):
    """
    Computing CQT with random shift as pitch shift augmentation

    Parameters
    ----------
    sr : int
        Sampling rate
    hop_length : float
        Hop length in seconds
    bins_per_octave : int
        Number of bins per octave
    num_octaves : int
        Number of octaves
    """

    def __init__(self, sr: int, hop_length: float, bins_per_octave: int, num_octaves: int) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.bins_per_semitone = bins_per_octave // 12
        self.num_octaves = num_octaves

        hop_frames = int(sr * hop_length)
        self.cqt_layer = features.cqt.CQT(
            sr=sr, hop_length=hop_frames, n_bins=(num_octaves + 1)*bins_per_octave, bins_per_octave=bins_per_octave)

    def forward(self, x: torch.Tensor, y: torch.Tensor, shift: bool = True):
        """
        Input
        ----------
        x : torch.Tensor
            Audio signal, shape (batch, num_samples)
        y : torch.Tensor
            Labels, shape (batch, num_frames)
        shift : bool
            Whether to shift the CQT spectrogram, default is True

        Output
        ----------
        s : torch.Tensor
            CQT spectrogram, shape (batch, 1, num_frames, num_bins)
        y : torch.Tensor
            Labels, shape (batch, num_frames)
        """

        with torch.no_grad():
            s = self.cqt_layer(x)
            s = torch.log(s + 1e-7)

            shift = torch.randint(low=-4, high=7+1, size=(len(x), 1)) if shift else torch.zeros((len(x), 1), dtype=torch.int64)

            s = s[torch.arange(len(s))[:, None], self.bins_per_semitone * (4 + shift) + torch.arange(self.num_octaves * self.bins_per_octave)]
            s = (s - torch.mean(s, dim=(1, 2), keepdim=True)) / torch.std(s, dim=(1, 2), keepdim=True)

            shift = shift.to(x.device)
            y = torch.where(y != -1, (y - shift * 2) % len(INDEX_TO_KEY), y)

            s = s.unsqueeze(dim=1)
        return s, y
