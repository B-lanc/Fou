import glob
import os


def load_xy(path, y=["vocals", "bass", "drums", "other"]):
    """loads the dataset with xy directory form, meant to be used for non shuffled dataloader.

    will only be used for stage 1 training
    xy directory refers to directory format where the inner directories are named after the track titles, and the files inside named after the instruments
    something like musdb18hq dataset directory (train / test directory inside musdb18hq dataset)
    x refers to the track titles, y refers to the instruments
    outer directory contains directories (inner) with the track title as its name
    inner directory contains multiple wav files, each with the name of the instruments (y)

    args:
        path: path to the outer directory
        y: list of instruments
    returns:
        samples: list of dictionaries ([{"vocals" : "path_to_vocals", "bass" : "path_to_bass"...}, {}, {}...])
    """
    tracks = glob.glog(os.path.join(path, "*"))
    samples = list()

    for track_folder in sorted(tracks):
        track = dict()
        for stem in y:
            audio_path = os.path.join(track_folder, stem + ".wav")
            track[stem] = audio_path
        samples.append(track)

    return samples


def load_yx(path, y=["vocals", "bass", "drums", "other"]):
    """loads the dataset with yx directory form, meant to be used for shuffled dataloader.

    yx directory refers to directory format where the inner directories are named after the instruments, and the files inside named after track titles
    x refers to the track titles, y refers to the instruments
    outer directory contains directories (inner) with the instruments (y) as its name
    inner directory contains multiple wav files, each with the name of the track titles

    args:
        path: path to the outer directory
        y: list of instruments
    returns:
        samples: dictionaries of lists ({"vocals" : ["path_to_vocals1", "path_to_vocals2"...], "bass" : []...})
    """
    samples = dict()

    for inst in y:
        samples[inst] = sorted(glob.glob(os.path.join(path, inst, "*")))

    return samples


def xy_to_yx(samples):
    """Converts the xy format to yx format

    The yx format will be used in this codebase and just generally here
    This is used to make loading the data from hdf files simple and consistent from both shuffle and non shuffled dataloaders

    args:
        samples: the output of load_xy
    returns:
        yx: the same thing, but in yx form
    """
    keys = samples[0].keys()
    yx = {key: [] for key in keys}
    for track in samples:
        for key in keys:
            yx[key].append(track[key])
    return yx
