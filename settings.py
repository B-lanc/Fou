import os

dataset_dir = "/dataset/"
checkpoint_dir = "/saves/"
sr = 44100

hdf_dir = "./hdf"
test_hdf = os.path.join(hdf_dir, "test.hdf5")
val_hdf = os.path.join(hdf_dir, "val.hdf5")
normal_train_hdf = os.path.join(hdf_dir, "normal_train.hdf5")
shuffle_train_hdf = os.path.join(hdf_dir, "shuffle_train.hdf5")
