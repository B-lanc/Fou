import numpy as np
import h5py
from torch.utils.data import Dataset
from sortedcontainers import SortedList

import math
import random


class NormalDataset(Dataset):
    """for testing and validation purposes"""

    def __init__(self, hdf_path, input_length, output_length, random_hops=False):
        super(NormalDataset, self).__init__()
        self.hdf = hdf_path
        self.input_length = input_length
        self.output_length = output_length
        self.diff = (input_length - output_length) // 2
        if random_hops:
            self.random_hops = output_length // 2
        else:
            self.random_hops = False

        lengths = []
        with h5py.File(self.hdf, "r") as f:
            num_tracks = len(f["vocals"])
            for i in range(num_tracks):
                lengths.append(math.ceil(len(f["vocals"][f"{i}"]) / self.output_length))
        self.starts = SortedList(np.cumsum(lengths))
        self.length = self.starts[-1]

    def __getitem__(self, index):
        pad_front = 0
        pad_back = 0

        track_idx = self.starts.bisect_right(index)
        if track_idx > 0:
            index = index - self.starts[track_idx - 1]

        with h5py.File(self.hdf, "r") as f:
            tl = len(f["vocals"][f"{track_idx}"])

            start_pos = index * self.output_length - self.diff
            if self.random_hops:
                start_pos = start_pos + random.randint(
                    -self.random_hops, self.random_hops
                )
            if start_pos > tl:
                start_pos = tl - start_pos

            end_pos = start_pos + self.input_length
            if start_pos < 0:
                pad_front = abs(start_pos)
                start_pos = 0

            if end_pos > tl:
                pad_back = end_pos - tl
                end_pos = tl
            vocals = f["vocals"][f"{track_idx}"][start_pos:end_pos]
            drums = f["drums"][f"{track_idx}"][start_pos:end_pos]
            bass = f["bass"][f"{track_idx}"][start_pos:end_pos]
            other = f["other"][f"{track_idx}"][start_pos:end_pos]

        mix = drums + bass + other + vocals
        if pad_back or pad_front:
            vocals = np.pad(vocals, (pad_front, pad_back))
            mix = np.pad(mix, (pad_front, pad_back))

        return mix, vocals[self.diff : self.input_length - self.diff]

    def __len__(self):
        return self.length


class NormalShuffleDataset(Dataset):
    """for basic shuffling with predetermined sources"""

    def __init__(
        self,
        hdf_path,
        input_length,
        output_length,
        n_insts=[1, 1, 1, 1],
        random_hops=False,
    ):
        super(NormalShuffleDataset, self).__init__()
        self.hdf = hdf_path
        self.input_length = input_length
        self.output_length = output_length
        self.diff = (input_length - output_length) // 2
        self.n_vocals = n_insts[0]
        self.n_drums = n_insts[1]
        self.n_bass = n_insts[2]
        self.n_other = n_insts[3]
        if random_hops:
            self.random_hops = output_length // 2
        else:
            self.random_hops = False

        lengths = []
        with h5py.File(self.hdf, "r") as f:
            num_tracks = len(f["vocals"])
            for i in range(num_tracks):
                lengths.append(math.ceil(len(f["vocals"][f"{i}"]) / self.output_length))
        self.starts = SortedList(np.cumsum(lengths))
        self.length = self.starts[-1]

        self.indexes_vocals = [
            [i for i in range(self.length)] for _ in range(self.n_vocals)
        ]
        self.indexes_drums = [
            [i for i in range(self.length)] for _ in range(self.n_drums)
        ]
        self.indexes_bass = [
            [i for i in range(self.length)] for _ in range(self.n_bass)
        ]
        self.indexes_other = [
            [i for i in range(self.length)] for _ in range(self.n_other)
        ]

    def getitem(self, index, inst):
        pad_front = 0
        pad_back = 0

        track_idx = self.starts.bisect_right(index)
        if track_idx > 0:
            index = index - self.starts[track_idx - 1]

        with h5py.File(self.hdf, "r") as f:
            tl = len(f[inst][f"{track_idx}"])

            start_pos = index * self.output_length - self.diff
            if self.random_hops:
                start_pos = start_pos + random.randint(
                    -self.random_hops, self.random_hops
                )
            if start_pos > tl:
                start_pos = tl - start_pos

            end_pos = start_pos + self.input_length
            if start_pos < 0:
                pad_front = abs(start_pos)
                start_pos = 0

            if end_pos > tl:
                pad_back = end_pos - tl
                end_pos = tl
            audio = f[inst][f"{track_idx}"][start_pos:end_pos]
        audio = np.pad(audio, (pad_front, pad_back))
        return audio

    def __getitem__(self, index):
        vocals = False
        drums = False
        bass = False
        other = False

        for i in range(self.n_vocals):
            if vocals is False:
                vocals = self.getitem(self.indexes_vocals[i][index], "vocals")
            else:
                vocals += self.getitem(self.indexes_vocals[i][index], "vocals")
        for i in range(self.n_drums):
            if drums is False:
                drums = self.getitem(self.indexes_drums[i][index], "drums")
            else:
                drums += self.getitem(self.indexes_drums[i][index], "drums")
        for i in range(self.n_bass):
            if bass is False:
                bass = self.getitem(self.indexes_bass[i][index], "bass")
            else:
                bass += self.getitem(self.indexes_bass[i][index], "bass")
        for i in range(self.n_other):
            if other is False:
                other = self.getitem(self.indexes_other[i][index], "other")
            else:
                other += self.getitem(self.indexes_other[i][index], "other")

        mix = drums + bass + other + vocals

        return mix, vocals[self.diff : self.input_length - self.diff]

    def __len__(self):
        return self.length

    def shuffle(self):
        for li in self.indexes_vocals:
            random.shuffle(li)
        for li in self.indexes_drums:
            random.shuffle(li)
        for li in self.indexes_bass:
            random.shuffle(li)
        for li in self.indexes_other:
            random.shuffle(li)


class BinaryShuffleDataset(Dataset):
    """for instead of using predetermined sources, just use vocals and noise"""

    def __init__(
        self,
        hdf_path,
        input_length,
        output_length,
        n_vox=1,
        n_noi=3,
        alpha_vox=1,
        alpha_noi=1,
        random_hops=True,
    ):
        super(BinaryShuffleDataset, self).__init__()
        self.hdf = hdf_path
        self.input_length = input_length
        self.output_length = output_length
        self.diff = (input_length - output_length) // 2
        self.n_vox = n_vox
        self.n_noi = n_noi
        self.a_vox = alpha_vox
        self.a_noi = alpha_noi
        if random_hops:
            self.random_hops = output_length // 2
        else:
            self.random_hops = False

        lengths_vox = []
        lengths_noi = []
        with h5py.File(self.hdf, "r") as f:
            num_vox = len(f["vocals"])
            for i in range(num_vox):
                lengths_vox.append(
                    math.ceil(len(f["vocals"][f"{i}"]) / self.output_length)
                )
            num_noi = len(f["noise"])
            for i in range(num_noi):
                lengths_noi.append(
                    math.ceil(len(f["noise"][f"{i}"]) / self.output_length)
                )
        self.starts_vox = SortedList(np.cumsum(lengths_vox))
        self.starts_noi = SortedList(np.cumsum(lengths_noi))
        self.length_vox = self.starts_vox[-1]
        self.length_noi = self.starts_noi[-1]

        self.indexes_vox = [
            [i for i in range(self.length_vox)] for _ in range(self.n_vox)
        ]
        self.indexes_noi = [
            [i for i in range(self.length_noi)] for _ in range(self.n_noi)
        ]

    def getitem(self, index, inst):
        pad_front = 0
        pad_back = 0

        if inst == "vocals":
            track_idx = self.starts_vox.bisect_right(index)
            if track_idx > 0:
                index = index - self.starts_vox[track_idx - 1]
        else:
            track_idx = self.starts_noi.bisect_right(index)
            if track_idx > 0:
                index = index - self.starts_noi[track_idx - 1]

        with h5py.File(self.hdf, "r") as f:
            tl = len(f[inst][f"{track_idx}"])

            start_pos = index * self.output_length - self.diff
            if self.random_hops:
                start_pos = start_pos + random.randint(
                    -self.random_hops, self.random_hops
                )
            if start_pos > tl:
                start_pos = tl - start_pos

            end_pos = start_pos + self.input_length
            if start_pos < 0:
                pad_front = abs(start_pos)
                start_pos = 0

            if end_pos > tl:
                pad_back = end_pos - tl
                end_pos = tl
            audio = f[inst][f"{track_idx}"][start_pos:end_pos]
        audio = np.pad(audio, (pad_front, pad_back))
        return audio

    def __getitem__(self, index):
        vocals = False
        noise = False

        for i in range(self.n_vox):
            if vocals is False:
                vocals = self.getitem(self.indexes_vox[i][index], "vocals")
            else:
                if random.random() < self.a_vox:
                    vocals += self.getitem(self.indexes_vox[i][index], "vocals")
        for i in range(self.n_noi):
            if noise is False:
                noise = self.getitem(self.indexes_noi[i][index], "noise")
            else:
                if random.random() < self.a_noi:
                    noise += self.getitem(self.indexes_noi[i][index], "noise")

        mix = vocals + noise

        return mix, vocals[self.diff : self.input_length - self.diff]

    def __len__(self):
        return self.length_vox

    def shuffle(self):
        for li in self.indexes_vox:
            random.shuffle(li)
        for li in self.indexes_noi:
            random.shuffle(li)
