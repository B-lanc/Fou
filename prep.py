import numpy as np
import librosa
import h5py

import settings

import glob
import os

"""
Script to write the hdf files
"""


def zero_index(filt, cons_zeros):
    """
    gets index of sequence of zeros that is longer than cons_zeros
    returns 2D numpy array acting as a list of lists, with shape (-1, 2), the 2 refers to the start and end of 0 sequence
    """
    iszero = np.concatenate(([0], np.equal(filt, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    inde = (absdiff == 1).nonzero()[0].reshape(-1, 2)
    inde = inde[(inde[:, 1] - inde[:, 0]) >= cons_zeros]
    return inde


def remove_zeros(filt, cons_zeros):
    """
    params:
      filt : 1D numpy array
      cons_zeros : number of consecutive zeros
    """
    _filt = np.trim_zeros(filt)
    if cons_zeros > 0:
        if _filt.any():
            indexes = zero_index(_filt, cons_zeros)
            for i in reversed(indexes):
                _filt = np.delete(_filt, np.s_[i[0] + 1 : i[1] + 1])
    return _filt


if __name__ == "__main__":
    train_val_set = glob.glob(os.path.join(settings.dataset_dir, "train", "*"))
    test_set = glob.glob(os.path.join(settings.dataset_dir, "test", "*"))
    train_set = train_val_set[:75]
    val_set = train_val_set[75:]
    CONSECUTIVE_ZEROS = 22050

    # preparing hdf
    if not os.path.exists(settings.hdf_dir):
        os.makedirs(settings.hdf_dir)
    test_path = os.path.join(settings.hdf_dir, "test.hdf5")
    val_path = os.path.join(settings.hdf_dir, "val.hdf5")
    normal_train_path = os.path.join(settings.hdf_dir, "normal_train.hdf5")
    shuffle_train_path = os.path.join(settings.hdf_dir, "shuffle_train.hdf5")
    ###Validation, Test, and normal training hdf
    # normal hdf file for musdb18hq, separated into [vocals, drums, bass, other], used for validation, testing, and baseline comparison
    # the file has groups of vocals, drums, bass, other, each group has datasets 1, 2, 3, ...
    # every group has the same length
    with h5py.File(test_path, "w") as f:
        vox = f.create_group("vocals")
        dru = f.create_group("drums")
        bas = f.create_group("bass")
        oth = f.create_group("other")

        for idx, track in enumerate(test_set):
            vocals, sr = librosa.load(
                os.path.join(track, "vocals.wav"), mono=True, sr=settings.sr
            )
            drums, sr = librosa.load(
                os.path.join(track, "drums.wav"), mono=True, sr=settings.sr
            )
            bass, sr = librosa.load(
                os.path.join(track, "bass.wav"), mono=True, sr=settings.sr
            )
            other, sr = librosa.load(
                os.path.join(track, "other.wav"), mono=True, sr=settings.sr
            )

            vox.create_dataset(
                f"{idx}", shape=vocals.shape, dtype=vocals.dtype, data=vocals
            )
            dru.create_dataset(
                f"{idx}", shape=drums.shape, dtype=drums.dtype, data=drums
            )
            bas.create_dataset(f"{idx}", shape=bass.shape, dtype=bass.dtype, data=bass)
            oth.create_dataset(
                f"{idx}", shape=other.shape, dtype=other.dtype, data=other
            )

    with h5py.File(val_path, "w") as f:
        vox = f.create_group("vocals")
        dru = f.create_group("drums")
        bas = f.create_group("bass")
        oth = f.create_group("other")

        for idx, track in enumerate(val_set):
            vocals, sr = librosa.load(
                os.path.join(track, "vocals.wav"), mono=True, sr=settings.sr
            )
            drums, sr = librosa.load(
                os.path.join(track, "drums.wav"), mono=True, sr=settings.sr
            )
            bass, sr = librosa.load(
                os.path.join(track, "bass.wav"), mono=True, sr=settings.sr
            )
            other, sr = librosa.load(
                os.path.join(track, "other.wav"), mono=True, sr=settings.sr
            )

            vox.create_dataset(
                f"{idx}", shape=vocals.shape, dtype=vocals.dtype, data=vocals
            )
            dru.create_dataset(
                f"{idx}", shape=drums.shape, dtype=drums.dtype, data=drums
            )
            bas.create_dataset(f"{idx}", shape=bass.shape, dtype=bass.dtype, data=bass)
            oth.create_dataset(
                f"{idx}", shape=other.shape, dtype=other.dtype, data=other
            )

    with h5py.File(normal_train_path, "w") as f:
        vox = f.create_group("vocals")
        dru = f.create_group("drums")
        bas = f.create_group("bass")
        oth = f.create_group("other")

        for idx, track in enumerate(train_set):
            vocals, sr = librosa.load(
                os.path.join(track, "vocals.wav"), mono=True, sr=settings.sr
            )
            drums, sr = librosa.load(
                os.path.join(track, "drums.wav"), mono=True, sr=settings.sr
            )
            bass, sr = librosa.load(
                os.path.join(track, "bass.wav"), mono=True, sr=settings.sr
            )
            other, sr = librosa.load(
                os.path.join(track, "other.wav"), mono=True, sr=settings.sr
            )

            vox.create_dataset(
                f"{idx}", shape=vocals.shape, dtype=vocals.dtype, data=vocals
            )
            dru.create_dataset(
                f"{idx}", shape=drums.shape, dtype=drums.dtype, data=drums
            )
            bas.create_dataset(f"{idx}", shape=bass.shape, dtype=bass.dtype, data=bass)
            oth.create_dataset(
                f"{idx}", shape=other.shape, dtype=other.dtype, data=other
            )

    ###Shuffle training hdf
    # binary hdf file with vocals and noise
    # the file has groups of vocals and noise, each group has datasets 1, 2, 3, ...
    # the groups does not have the same length
    with h5py.File(shuffle_train_path, "w") as f:
        vox = f.create_group("vocals")
        noi = f.create_group("noise")

        for idx, track in enumerate(train_set):
            vocals, sr = librosa.load(
                os.path.join(track, "vocals.wav"), mono=True, sr=settings.sr
            )
            drums, sr = librosa.load(
                os.path.join(track, "drums.wav"), mono=True, sr=settings.sr
            )
            bass, sr = librosa.load(
                os.path.join(track, "bass.wav"), mono=True, sr=settings.sr
            )
            other, sr = librosa.load(
                os.path.join(track, "other.wav"), mono=True, sr=settings.sr
            )

            # removing consecutive zeros and then normalizing audio loudness (normalizing only happens downwards)
            vocals = remove_zeros(vocals, CONSECUTIVE_ZEROS)
            drums = remove_zeros(drums, CONSECUTIVE_ZEROS)
            bass = remove_zeros(bass, CONSECUTIVE_ZEROS)
            other = remove_zeros(other, CONSECUTIVE_ZEROS)

            vox.create_dataset(
                f"{idx}", shape=vocals.shape, dtype=vocals.dtype, data=vocals
            )
            noi.create_dataset(
                f"{idx*3}", shape=drums.shape, dtype=drums.dtype, data=drums
            )
            noi.create_dataset(
                f"{idx*3 + 1}", shape=bass.shape, dtype=bass.dtype, data=bass
            )
            noi.create_dataset(
                f"{idx*3 + 2}", shape=other.shape, dtype=other.dtype, data=other
            )
