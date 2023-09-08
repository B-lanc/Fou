import torch
import torch.nn as nn

from .UNet import UNet2D, MaskUNet2D
from .STFT import STFT, ISTFT
from .util import frames_to_samples, crop_center2d


class Fou(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        n_fft,
        hop_size,
        min_outputs,
        autopadding=True,
        hidden_kernel_size=None,
        scaled=False,
        activation="relu",
        norm="gn",
        dropout_rate=0.1,
    ):
        assert channels[0] == 2
        super(Fou, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.UNet = UNet2D(
            channels,
            kernel_size,
            stride,
            autopadding,
            hidden_kernel_size,
            activation=activation,
            norm=norm,
            dropout_rate=dropout_rate,
        )
        (bins_input, bins_output), (
            frames_input,
            frames_output,
        ) = self.UNet.get_io_sizes(min_outputs)
        self.input_samples = (
            frames_to_samples(frames_input, self.n_fft, self.hop_size)
        ) - self.n_fft  # also because of padding
        self.output_samples = (
            frames_to_samples(frames_output, self.n_fft, self.hop_size)
        ) - self.n_fft  # as it is being cropped again after istft

        self.stft = STFT(self.n_fft, self.hop_size, True, scaled=scaled)
        self.istft = ISTFT(self.n_fft, self.hop_size, True, scaled=scaled)

    def forward(self, x):
        out = self.stft(x)  # (real, imag) (bs, 1, n_frames, n_fft//2 + 1)
        out = torch.cat(out, dim=1)  # (bs, 2, n_frames, n_fft//2 + 1)

        out = out.transpose(2, 3)  # (bs, 2, bins_input, frames_input)
        out = self.UNet(out)  # (bs, 2, bins_output, frames_output)

        real = out[:, 0, :, :]
        imag = out[:, 1, :, :]
        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)  # (bs, 1, n_frames, bins)

        out = self.istft(real, imag, None)
        return out


class MaskFou(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        n_fft,
        hop_size,
        min_outputs,
        autopadding=True,
        hidden_kernel_size=None,
        scaled=False,
        activation="relu",
        norm="gn",
        dropout_rate=0.1,
    ):
        assert channels[0] == 2
        super(MaskFou, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.MaskUNet = MaskUNet2D(
            channels,
            kernel_size,
            stride,
            autopadding,
            hidden_kernel_size,
            activation=activation,
            norm=norm,
            dropout_rate=dropout_rate,
        )
        (bins_input, bins_output), (
            frames_input,
            frames_output,
        ) = self.MaskUNet.get_io_sizes(min_outputs)
        self.input_samples = (
            frames_to_samples(frames_input, self.n_fft, self.hop_size)
        ) - self.n_fft  # also because of padding
        self.output_samples = (
            frames_to_samples(frames_output, self.n_fft, self.hop_size)
        ) - self.n_fft  # as it is being cropped again after istft

        self.stft = STFT(self.n_fft, self.hop_size, True, scaled=scaled)
        self.istft = ISTFT(self.n_fft, self.hop_size, True, scaled=scaled)

    def forward(self, x):
        out = self.stft(x)  # (real, imag) (bs, 1, n_frames, n_fft//2 + 1)
        out = torch.cat(out, dim=1)  # (bs, 2, n_frames, n_fft//2 + 1)

        out = out.transpose(2, 3)  # (bs, 2, bins_input, frames_input)
        out = self.MaskUNet(out)  # (bs, 2, bins_output, frames_output)

        real = out[:, 0, :, :]
        imag = out[:, 1, :, :]
        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)  # (bs, 1, n_frames, bins)

        out = self.istft(real, imag, None)
        return out