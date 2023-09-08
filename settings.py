import os

cuda = True

dataset_dir = "/dataset/"
checkpoint_dir = "/saves/"
sr = 44100

hdf_dir = "./hdf"
test_hdf = os.path.join(hdf_dir, "test.hdf5")
val_hdf = os.path.join(hdf_dir, "val.hdf5")
normal_train_hdf = os.path.join(hdf_dir, "normal_train.hdf5")
shuffle_train_hdf = os.path.join(hdf_dir, "shuffle_train.hdf5")

training_tag = "MaskFou"
n_vocals = 2
n_noise = 6
alpha_vox = 0.8 #chance of additional vocals
alpha_noi = 0.85 #chance of additional noise
channels = [2, 32, 64, 128]
output_min = [256, 150]  # number of minimum output bins and frames
kernel = 5
stride = 2
n_fft = 2048
hop_size = 512
norm = "bn"
drop_rate = 0.1
scaled = False  # STFT settings
activation = "tanh"
epoch_patience = 40
lr = 1e-4
batch_size = 32
loss = "MSE"