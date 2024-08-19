import os
from itertools import product

import soundfile as sf

import numpy as np
import pandas as pd

import torch.utils.data as Data


ALL_VERSIONS = ["HU33", "SC06", "AL98", "FI55", "FI66", "FI80", "OL06", "QU98", "TR99"]
ALL_SONGS = ["{:02d}".format(x) for x in range(1, 24+1)]

INDEX_TO_KEY = [
    "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min", "C:maj", "C:min",
    "C#:maj", "C#:min", "D:maj", "D:min", "D#:maj", "D#:min", "E:maj", "E:min",
    "F:maj", "F:min", "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min"
]
KEY_TO_INDEX = {key: i for i, key in enumerate(INDEX_TO_KEY)}


def get_split_list(split="neither"):
    train_list, valid_list, test_list = [], [], []

    if split == "version":
        versions_split = [np.roll(ALL_VERSIONS, i) for i in [0, 3, 6]]
        for versions in versions_split:
            train_list.append(["Schubert_D911-{}_{}".format(song, version) for song in ALL_SONGS for version in versions[:5]])
            valid_list.append(["Schubert_D911-{}_{}".format(song, version) for song in ALL_SONGS for version in versions[5:6]])
            test_list.append(["Schubert_D911-{}_{}".format(song, version) for song in ALL_SONGS for version in versions[6:]])
            
    elif split == "song":
        songs_split = [np.roll(ALL_SONGS, i) for i in [0, 8, 16]]
        for songs in songs_split:
            train_list.append(["Schubert_D911-{}_{}".format(song, version) for version in ALL_VERSIONS for song in songs[:13]])
            valid_list.append(["Schubert_D911-{}_{}".format(song, version) for version in ALL_VERSIONS for song in songs[13:16]])
            test_list.append(["Schubert_D911-{}_{}".format(song, version) for version in ALL_VERSIONS for song in songs[16:]])

    elif split == "neither":
        versions_split = [np.roll(ALL_VERSIONS, i) for i in [0, 3, 6]]
        songs_split = [np.roll(ALL_SONGS, i) for i in [0, 3, 6, 9, 12, 15, 18, 21]]
        for versions, songs in product(versions_split, songs_split):
            train_list.append(["Schubert_D911-{}_{}".format(song, version) for version in versions[:4] for song in songs[:19]])
            valid_list.append(["Schubert_D911-{}_{}".format(song, version) for version in versions[4:6] for song in songs[19:21]])
            test_list.append(["Schubert_D911-{}_{}".format(song, version) for version in versions[6:] for song in songs[21:]])
        
    return train_list, valid_list, test_list


class SingleSongDataset(Data.Dataset):
    def __init__(
        self,
        audio_path,
        label_path,

        sr=22050,

        seg_length=10,
        seg_hop_length=2,

        hop_length=0.2,

        shift=True
    ) -> None:
        super().__init__()

        self.audio_path = audio_path
        self.label_path = label_path

        self.sr = sr

        self.seg_length = seg_length
        self.seg_hop_length = seg_hop_length
        self.seg_frames = int(sr * self.seg_length) if seg_length > 0 else -1
        self.seg_hop_frames = int(sr * seg_hop_length)

        self.hop_length = hop_length
        self.hop_frames = int(hop_length * sr)

        self.shift = shift
        self.df = pd.read_csv(self.label_path, sep=";")

        audio, sr = sf.read(self.audio_path)
        assert sr == self.sr
        self.length = (len(audio) - self.seg_frames) // self.seg_hop_frames if self.seg_length > 0 else 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.seg_length > 0:
            start_frame = index * self.seg_hop_frames
            audio, sr = sf.read(self.audio_path, start=start_frame, frames=self.seg_frames, dtype=np.float32)
            frame = np.arange(start_frame, start_frame+self.seg_frames+1, self.hop_frames)  # same shape as cqt spectrogram
        else:
            audio, sr = sf.read(self.audio_path, dtype=np.float32)
            frame = np.arange(0, len(audio) + 1, self.hop_frames)
        time_frame = frame / sr

        labels = -1 * np.ones(len(time_frame), dtype=np.int64)
        for _, row in self.df.iterrows():
            labels[(time_frame > row["start"]) & (time_frame + self.hop_frames / sr < row["end"])] = KEY_TO_INDEX[row["key"]]

        return dict(x=audio, y=labels)


def get_SWD_dataloader(
    dataset_path,
    piece_list,

    seg_length=10,
    seg_hop_length=2,

    hop_length=0.2,

    batch_size=32,
    shuffle=True,
    num_workers=8
):
    audio_folder = os.path.join(dataset_path, "01_RawData/audio_wav")
    label_folder = os.path.join(dataset_path, "02_Annotations/ann_audio_localkey-ann3")
    
    dataset_list = []
    for piece in piece_list:
        dataset = SingleSongDataset(
            audio_path=os.path.join(audio_folder, "{}.wav".format(piece)),
            label_path=os.path.join(label_folder, "{}.csv".format(piece)),
            sr=22050,
            seg_length=seg_length,
            seg_hop_length=seg_hop_length,
            hop_length=hop_length
        )
        dataset_list.append(dataset)
    
    dataloader = Data.DataLoader(
        Data.ConcatDataset(dataset_list), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
