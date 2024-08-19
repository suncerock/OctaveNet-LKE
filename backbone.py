import torch
import torch.nn as nn


NUM_CLASSES = 24

class OctaveNet(nn.Module):
    def __init__(
        self,
        channel=16,
        bins_per_octave=24,
        num_octaves=6,
        dropout=0.3,
        
        chroma=True,
        octave=True,
        fuse=True
    ) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self.num_octaves = num_octaves

        self.chroma = chroma
        self.octave = octave
        self.fuse = fuse

        if self.octave:
            self.conv_octave = nn.Sequential(
                nn.Conv2d(self.bins_per_octave, channel, 1),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            )

            self.agg_octave = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(self.num_octaves, 1)),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            )

            self.rnn_octave = nn.LSTM(channel, channel, num_layers=1, batch_first=True, bidirectional=True)

        if self.chroma:
            self.conv_chroma = nn.Sequential(
                nn.Conv2d(self.num_octaves, channel, 1),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            )

            self.agg_chroma = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=(self.bins_per_octave, 1)),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            )

            self.rnn_chroma = nn.LSTM(channel, channel, num_layers=1, batch_first=True, bidirectional=True)

        if self.fuse:
            self.fuse = nn.Sequential(
                nn.Linear(channel * 2 * int(chroma + octave), channel * 2),
                nn.ReLU(),
                nn.Linear(channel * 2, channel * 2),
            )
            self.rnn_combine = nn.LSTM(channel * 2, channel * 2, num_layers=1, batch_first=True, bidirectional=True)
        
        self.linear = nn.Linear(channel * 4, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        assert x.shape[2] == self.bins_per_octave * self.num_octaves

        x = x.reshape(x.shape[0], self.num_octaves, self.bins_per_octave, x.shape[-1])

        x_chroma = x
        x_octave = x.transpose(1, 2)

        if self.octave:
            x_octave = self.conv_octave(x_octave)
            x_octave = self.agg_octave(x_octave)
            x_octave = x_octave.squeeze(dim=2).transpose(-1, -2)
            x_octave = self.dropout(x_octave)

            x_octave, _ = self.rnn_octave(x_octave)
            x_octave = self.dropout(x_octave)

        if self.chroma:
            x_chroma = self.conv_chroma(x_chroma)
            x_chroma = self.agg_chroma(x_chroma)
            x_chroma = x_chroma.squeeze(dim=2).transpose(-1, -2)
            x_chroma = self.dropout(x_chroma)

            x_chroma, _ = self.rnn_chroma(x_chroma)
            x_chroma = self.dropout(x_chroma)

        if self.octave and self.chroma:
            x = torch.concat([x_octave, x_chroma], dim=-1)
        else:
            x = x_octave if self.octave else x_chroma

        if self.fuse:
            x = self.dropout(self.fuse(x))
            x, _ = self.rnn_combine(x)

        x = self.dropout(x)
        x = self.linear(x).transpose(-1, -2)

        return x
