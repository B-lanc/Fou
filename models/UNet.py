import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import conv_input, conv_output, crop_center2d

from functools import partial


class UNet2D(nn.Module):
    """
    UNet template using Conv2d
    """

    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        autopadding=True,
        hidden_kernel_size=None,
        activation="relu",
        norm="gn",
        dropout_rate=0.1,
    ):
        """
        params:
            channels->[int] list of integers as channel input and output (at least length should be 3)
            kernel_size->int the kernel size for down/upsampling blocks (should be odd)
            stride->int the stride for down/upsampling blocks (should be even)
            autopadding->bool whether to use automatical padding for hidden blocks, True would mean the output is exactly the same, False would reduce the output size
            hidden_kernel_size->int|None the kernel size for hidden blocks (should be odd), if None, then the same as kernel size
        """
        super(UNet2D, self).__init__()
        self.channels = channels
        self.n_layers = len(channels) - 1
        if not hidden_kernel_size:
            hidden_kernel_size = kernel_size
        if autopadding:
            autopadding = hidden_kernel_size // 2
        else:
            autopadding = 0

        if activation.lower() == "relu":
            self.activation = F.relu
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        elif activation.lower() == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = F.leaky_relu

        self.drop = nn.Dropout(dropout_rate)

        if norm.lower() == "bn":
            norm = nn.BatchNorm1d
        elif norm.lower() == "gn":
            norm = nn.GroupNorm
            norm = partial(norm, 8)
        elif norm.lower() == "none":
            norm = nn.Identity
        else:
            raise NotImplementedError

        self.dsh_blocks = nn.ModuleList()
        self.ds_blocks = nn.ModuleList()
        self.us_blocks = nn.ModuleList()
        self.ush_blocks = nn.ModuleList()
        self.dsh_norm = nn.ModuleList()
        self.ds_norm = nn.ModuleList()
        self.us_norm = nn.ModuleList()
        self.ush_norm = nn.ModuleList()

        self.bottle = nn.Conv2d(
            channels[-1],
            channels[-1],
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=autopadding,
        )  # hidden block

        # DS/US blocks
        for i in range(self.n_layers):
            self.dsh_blocks.append(
                nn.Conv2d(
                    channels[i],
                    channels[i],
                    kernel_size=hidden_kernel_size,
                    stride=1,
                    padding=autopadding,
                )
            )
            self.ds_blocks.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            self.us_blocks.append(
                nn.ConvTranspose2d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            self.ush_blocks.append(
                nn.Conv2d(
                    channels[i] * 2,
                    channels[i],
                    kernel_size=hidden_kernel_size,
                    stride=1,
                    padding=autopadding,
                )
            )
            self.dsh_norm.append(norm(channels[i]))
            self.ds_norm.append(norm(channels[i + 1]))
            self.ush_norm.append(norm(channels[i]))
            self.us_norm.append(norm(channels[i]))

    def forward(self, x):
        shortcut = []
        for ds, dsh, dsn, dshn in zip(
            self.ds_blocks, self.dsh_blocks, self.ds_norm, self.dsh_norm
        ):
            x = dshn(self.activation(dsh(x)))
            x = self.drop(x)
            shortcut.append(x)
            x = dsn(self.activation(ds(x)))
            x = self.drop(x)
        x = self.bottle(x)
        for i in reversed(range(self.n_layers)):
            x = self.us_norm[i](self.activation(self.us_blocks[i](x)))
            x = self.drop(x)
            x = torch.cat((x, crop_center2d(shortcut[i], x)), dim=1)
            x = self.ush_norm[i](self.activation(self.ush_blocks[i](x)))
            x = self.drop(x)
        return x

    def get_io_sizes(self, output_min):
        """
        finds input and output combination where output is higher than output_min, and the difference of shortcut to be even
        params:
            output_min:->[int]|int list of 2 integers as the minimum output
        """
        if isinstance(output_min, int):
            output_min = [output_min, output_min]
        bottle = 1
        while True:
            try:
                input_size, output_size = self.check_bottle(bottle, 0)
                assert input_size % 2 == 1
                assert output_size >= output_min[0]
                height = (input_size, output_size)
                break
            except AssertionError:
                bottle += 1
        bottle = 1
        while True:
            try:
                input_size, output_size = self.check_bottle(bottle, 1)
                assert output_size >= output_min[1]
                assert input_size % 2 == 1
                width = (input_size, output_size)
                return height, width
            except AssertionError:
                bottle += 1

    def check_bottle(self, bottle, ax):
        """bottle is the size after the bottleneck block"""
        pre = conv_input(
            bottle,
            self.bottle.kernel_size[ax],
            self.bottle.padding[ax],
            self.bottle.stride[ax],
        )
        post = bottle
        for i in reversed(range(self.n_layers)):
            pre = conv_input(
                pre,
                self.ds_blocks[i].kernel_size[ax],
                self.ds_blocks[i].padding[ax],
                self.ds_blocks[i].stride[ax],
            )
            post = conv_input(
                post,
                self.us_blocks[i].kernel_size[ax],
                self.us_blocks[i].padding[ax],
                self.us_blocks[i].stride[ax],
            )  # input instead of output because it's convtranspose

            assert (pre - post) % 2 == 0  # so can be cropped equally

            pre = conv_input(
                pre,
                self.dsh_blocks[i].kernel_size[ax],
                self.dsh_blocks[i].padding[ax],
                self.dsh_blocks[i].stride[ax],
            )
            post = conv_output(
                post,
                self.ush_blocks[i].kernel_size[ax],
                self.ush_blocks[i].padding[ax],
                self.ush_blocks[i].stride[ax],
            )

        return pre, post


class MaskUNet2D(nn.Module):
    """
    UNet template using Conv2d
    """

    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        autopadding=True,
        hidden_kernel_size=None,
        activation="relu",
        norm="gn",
        dropout_rate=0.1,
    ):
        """
        params:
            channels->[int] list of integers as channel input and output (at least length should be 3)
            kernel_size->int the kernel size for down/upsampling blocks (should be odd)
            stride->int the stride for down/upsampling blocks (should be even)
            autopadding->bool whether to use automatical padding for hidden blocks, True would mean the output is exactly the same, False would reduce the output size
            hidden_kernel_size->int|None the kernel size for hidden blocks (should be odd), if None, then the same as kernel size
        """
        super(MaskUNet2D, self).__init__()
        self.channels = channels
        self.n_layers = len(channels) - 1
        if not hidden_kernel_size:
            hidden_kernel_size = kernel_size
        if autopadding:
            autopadding = hidden_kernel_size // 2
        else:
            autopadding = 0

        if activation.lower() == "relu":
            self.activation = F.relu
        elif activation.lower() == "tanh":
            self.activation = torch.tanh
        elif activation.lower() == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = F.leaky_relu

        self.drop = nn.Dropout(dropout_rate)

        if norm.lower() == "bn":
            norm = nn.BatchNorm1d
        elif norm.lower() == "gn":
            norm = nn.GroupNorm
            norm = partial(norm, 8)
        elif norm.lower() == "none":
            norm = nn.Identity
        else:
            raise NotImplementedError

        self.dsh_blocks = nn.ModuleList()
        self.ds_blocks = nn.ModuleList()
        self.us_blocks = nn.ModuleList()
        self.ush_blocks = nn.ModuleList()
        self.dsh_norm = nn.ModuleList()
        self.ds_norm = nn.ModuleList()
        self.us_norm = nn.ModuleList()
        self.ush_norm = nn.ModuleList()

        self.bottle = nn.Conv2d(
            channels[-1],
            channels[-1],
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=autopadding,
        )  # hidden block

        # DS/US blocks
        for i in range(self.n_layers):
            self.dsh_blocks.append(
                nn.Conv2d(
                    channels[i],
                    channels[i],
                    kernel_size=hidden_kernel_size,
                    stride=1,
                    padding=autopadding,
                )
            )
            self.ds_blocks.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            self.us_blocks.append(
                nn.ConvTranspose2d(
                    channels[i + 1],
                    channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            self.ush_blocks.append(
                nn.Conv2d(
                    channels[i],
                    channels[i],
                    kernel_size=hidden_kernel_size,
                    stride=1,
                    padding=autopadding,
                )
            )
            self.dsh_norm.append(norm(channels[i]))
            self.ds_norm.append(norm(channels[i + 1]))
            self.us_norm.append(norm(channels[i]))
            self.ush_norm.append(norm(channels[i]))

    def forward(self, x):
        shortcut = []
        for ds, dsh, dsn, dshn in zip(
            self.ds_blocks, self.dsh_blocks, self.ds_norm, self.dsh_norm
        ):
            x = dshn(self.activation(dsh(x)))
            x = self.drop(x)
            shortcut.append(x)
            x = dsn(self.activation(ds(x)))
            x = self.drop(x)
        x = self.bottle(x)
        for i in reversed(range(self.n_layers)):
            x = self.us_norm[i](self.activation(self.us_blocks[i](x)))
            x = self.drop(x)
            x = x * crop_center2d(shortcut[i], x)
            x = self.ush_norm[i](self.activation(self.ush_blocks[i](x)))
            x = self.drop(x)
        return x

    def get_io_sizes(self, output_min):
        """
        finds input and output combination where output is higher than output_min, and the difference of shortcut to be even
        params:
            output_min:->[int]|int list of 2 integers as the minimum output
        """
        if isinstance(output_min, int):
            output_min = [output_min, output_min]
        bottle = 1
        while True:
            try:
                input_size, output_size = self.check_bottle(bottle, 0)
                assert input_size % 2 == 1
                assert output_size >= output_min[0]
                height = (input_size, output_size)
                break
            except AssertionError:
                bottle += 1
        bottle = 1
        while True:
            try:
                input_size, output_size = self.check_bottle(bottle, 1)
                assert output_size >= output_min[1]
                assert input_size % 2 == 1
                width = (input_size, output_size)
                return height, width
            except AssertionError:
                bottle += 1

    def check_bottle(self, bottle, ax):
        """bottle is the size after the bottleneck block"""
        pre = conv_input(
            bottle,
            self.bottle.kernel_size[ax],
            self.bottle.padding[ax],
            self.bottle.stride[ax],
        )
        post = bottle
        for i in reversed(range(self.n_layers)):
            pre = conv_input(
                pre,
                self.ds_blocks[i].kernel_size[ax],
                self.ds_blocks[i].padding[ax],
                self.ds_blocks[i].stride[ax],
            )
            post = conv_input(
                post,
                self.us_blocks[i].kernel_size[ax],
                self.us_blocks[i].padding[ax],
                self.us_blocks[i].stride[ax],
            )  # input instead of output because it's convtranspose

            assert (pre - post) % 2 == 0  # so can be cropped equally

            pre = conv_input(
                pre,
                self.dsh_blocks[i].kernel_size[ax],
                self.dsh_blocks[i].padding[ax],
                self.dsh_blocks[i].stride[ax],
            )
            post = conv_output(
                post,
                self.ush_blocks[i].kernel_size[ax],
                self.ush_blocks[i].padding[ax],
                self.ush_blocks[i].stride[ax],
            )

        return pre, post
