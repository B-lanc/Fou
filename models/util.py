def conv_output(size_in, kernel_size, padding=0, stride=1, dilation=1):
    return (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv_input(
    size_out, kernel_size, padding=0, stride=1, dilation=1, output_padding=0
):
    return (
        (size_out - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def crop_center(source, target):
    s = source.shape[2]
    t = target.shape[2]
    if s == t:
        return source
    assert (s - t) % 2 == 0
    return source[:, :, (s - t) // 2 : -((s - t) // 2)]


def crop_center2d(source, target):
    """cropping the source tensor to match target tensor"""
    if source.shape[2] == target.shape[2]:
        ax2 = [None, None]
    else:
        diff = (source.shape[2] - target.shape[2]) // 2
        ax2 = [diff, -diff]
    if source.shape[3] == target.shape[3]:
        ax3 = [None, None]
    else:
        diff = (source.shape[3] - target.shape[3]) // 2
        ax3 = [diff, -diff]
    return source[:, :, ax2[0] : ax2[1], ax3[0] : ax3[1]]


def frames_to_samples(frames, n_fft=2048, hop_size=512):
    return (frames - 1) * hop_size + n_fft
