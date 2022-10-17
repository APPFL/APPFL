def get_model():
    import torch.nn as nn
    def conv_block(in_planes, out_planes, stride=1, groups=1, dilation=1):
        return nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=17,
            stride=stride,
            padding=8,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
        ):
            super(BasicBlock, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            if groups != 1 or base_width != 64:
                raise ValueError("BasicBlock only supports groups=1 and base_width=64")
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv_block(inplanes, planes, stride)
            self.bn1 = norm_layer(inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv_block(planes, planes)
            self.bn2 = norm_layer(planes)
            self.dropout = nn.Dropout()
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            
            # BN > RELU > Dropout > Conv
            out = self.bn1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv1(out)
            # BN > RELU > Dropout > Conv
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            # Add skip
            out += identity

            return out

    def conv_subsumpling(in_planes, out_planes, stride=1):
        return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class EcgResNet34(nn.Module):
        def __init__(
            self,
            layers=(1, 5, 5, 5),
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            block=BasicBlock,
        ):
            super(EcgResNet34, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            self._norm_layer = norm_layer

            self.inplanes = 32
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError(
                    "replace_stride_with_dilation should be None "
                    "or a 3-element tuple, got {}".format(replace_stride_with_dilation),
                )
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = conv_block(12, self.inplanes, stride=1) # 12 leads
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
            )
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            # change to regression
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv_subsumpling(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    self.groups,
                    self.base_width,
                    previous_dilation,
                    norm_layer,
                ),
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    ),
                )

            return nn.Sequential(*layers)

        def forward(self, x):
            x = x.float()
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)

            return x
    return EcgResNet34