import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class STFT(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_size,
        freeze_parameter=True,
        padding=True,
        pad_mode="reflect",
        scaled=False,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = padding
        self.pad_mode = pad_mode  # reflect or constant
        self.scaled = scaled

        if not hop_size:
            self.hop_size = self.n_fft // 4

        hann = librosa.filters.get_window("hann", self.n_fft, fftbins=True)

        ohm = np.exp(-2 * np.pi * 1j / self.n_fft)
        w = np.matmul(
            np.arange(self.n_fft)[:, np.newaxis],
            np.arange(self.n_fft // 2 + 1)[np.newaxis, :],
        )
        self.W = np.power(ohm, w) * hann[:, np.newaxis]  # (n_fft, n_fft//2+1)

        self.conv_real = nn.Conv1d(
            in_channels=1,
            out_channels=self.n_fft // 2 + 1,
            kernel_size=self.n_fft,
            stride=self.hop_size,
            bias=False,
        )
        self.conv_imag = nn.Conv1d(
            in_channels=1,
            out_channels=self.n_fft // 2 + 1,
            kernel_size=self.n_fft,
            stride=self.hop_size,
            bias=False,
        )

        self.conv_real.weight.data = torch.Tensor(np.real(self.W).T)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W).T)[:, None, :]
        # (n_fft//2+1, 1, n_fft), Note... the shape of conv1d weight is (output_channel, input_channel, kernel_size)

        if freeze_parameter:
            for param in self.parameters():
                param.require_grad = False

    def forward(self, input):
        x = input[:, None, :]  # (bs, channels_in, data length)

        if self.padding:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        # (batch_size, channels->1, n_frames, n_fft//2+1)
        if self.scaled:
            real = real * 2 / self.n_fft
            imag = imag * 2 / self.n_fft

        return real, imag


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_size,
        freeze_parameter=True,
        padding=True,
        pad_mode="reflect",
        scaled=False,
    ):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = padding
        self.pad_mode = pad_mode  # reflect or constant

        if not hop_size:
            self.hop_size = self.n_fft // 4

        # Conv1d for calculating real and imag part
        hann = librosa.filters.get_window("hann", self.n_fft, fftbins=True)
        ohm = np.exp(2 * np.pi * 1j / self.n_fft)
        w = np.matmul(
            np.arange(self.n_fft)[:, np.newaxis],
            np.arange(self.n_fft)[np.newaxis, :],
        )  # (n_fft, n_fft)
        self.W = np.power(ohm, w) * hann[np.newaxis, :]  # (n_fft, n_fft)
        if scaled:
            self.W = self.W / 2
        else:
            self.W = self.W / self.n_fft

        self.conv_real = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.conv_imag = nn.Conv1d(
            in_channels=self.n_fft,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.conv_real.weight.data = torch.Tensor(np.real(self.W).T)[:, :, None]
        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W).T)[:, :, None]
        # (n_fft, n_fft, 1)

        # overlap add window to reconstruct time domain signal
        ola_window = librosa.util.normalize(hann, norm=None) ** 2
        ola_window = torch.Tensor(ola_window)
        self.register_buffer("ola_window", ola_window)

        if freeze_parameter:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, real_stft, imag_stft, length=None):
        """
        Shape of stft is (batch_size, channels->should be 1, n_frames, n_fft//2+1)
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        batch_size, _, n_frames, _ = real_stft.shape

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (bs, n_fft//2+1, n_frames)

        full_real_stft = torch.cat(
            (real_stft, torch.flip(real_stft[:, 1:-1, :], dims=[1])), dim=1
        )  # (bs, n_fft, n_frames)
        full_imag_stft = torch.cat(
            (imag_stft, -torch.flip(imag_stft[:, 1:-1, :], dims=[1])), dim=1
        )  # (bs, n_fft, n_frames)

        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)
        # (bs, n_fft, n_frames)

        # overlap
        output_samples = (n_frames - 1) * self.hop_size + self.n_fft
        y = F.fold(
            input=s_real,
            output_size=(1, output_samples),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_size),
        )  # (batch_size, 1, 1, audio_samples)
        y = y[:, 0, 0, :]  # (batch_size, audio_samples)

        window_matrix = self.ola_window[None, :, None].repeat(1, 1, n_frames)
        # (batch_size, win_length, n_frames)
        ifft_window_sum = F.fold(
            input=window_matrix,
            output_size=(1, output_samples),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_size),
        )  # (1, 1, 1, audio_samples)
        ifft_window_sum = ifft_window_sum.squeeze()
        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-11, np.inf)

        y = y / ifft_window_sum[None, :]
        if length:
            y = y[:, self.n_fft // 2 : self.n_fft // 2 + length]
        else:
            y = y[:, self.n_fft // 2 : -self.n_fft // 2]

        return y
