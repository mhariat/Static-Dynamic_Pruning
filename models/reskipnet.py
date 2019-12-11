from models.models import *
from models.gates import *
from utils.compute_flops import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if (stride != 1) | (inplanes != planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        self.additional_flops = residual.nelement()
        self.additional_parameters = 0
        out = self.relu(out)
        return out

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        self.conv2 = BasisLayer(self.conv2)
        if self.downsample is not None:
            self.downsample[0] = BasisLayer(self.downsample[0])

    def get_additional_flops(self):
        assert hasattr(self, 'additional_flops'), 'No additional_flops attr. Consider launching forward hook!'
        return self.additional_flops

    def get_additional_parameters(self):
        assert hasattr(self, 'additional_parameters'), 'No additional_parameters attr. Consider launching forward hook!'
        return self.additional_parameters


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=1, stride=1,
                               bias=False)
        self.conv3.show = True
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if (stride != 1) | (inplanes != self.expansion*planes):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=self.expansion*planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        self.additional_flops = residual.nelement()
        self.additional_parameters = 0
        out += residual
        out = self.relu(out)
        return out

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        self.conv2 = BasisLayer(self.conv2)
        self.conv3 = BasisLayer(self.conv3)
        if self.downsample is not None:
            self.downsample[0] = BasisLayer(self.downsample[0])

    def get_additional_flops(self):
        assert hasattr(self, 'additional_flops'), 'No additional_flops attr. Consider launching forward hook!'
        return self.additional_flops

    def get_additional_parameters(self):
        assert hasattr(self, 'additional_parameters'), 'No additional_parameters attr. Consider launching forward hook!'
        return self.additional_parameters


class GateLayer(nn.Module):
    def __init__(self, embed_dim, pool_size, inplanes):
        super(GateLayer, self).__init__()
        self.inplanes = inplanes
        self.pool_size = pool_size
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels=self.inplanes, out_channels=self.embed_dim, kernel_size=1, stride=1)
        self.conv.not_interesting = True

    def forward(self, x):
        x = nn.AvgPool2d(self.pool_size)(x)
        x = self.conv(x)
        return x


model_layers = {
    'resnet18': (BasicBlock, [2, 2, 2, 2]),
    'resnet34': (BasicBlock, [3, 4, 6, 3]),
    'resnet50': (Bottleneck, [3, 4, 6, 3]),
    'resnet101': (Bottleneck, [3, 4, 23, 3]),
    'resnet152': (Bottleneck, [3, 8, 36, 3])
}


class _ReSkipNet(Models):
    def __init__(self, depth, num_classes=1000, embed_dim=10, hidden_dim=10, rl_mode=False, inference_mode=False):
        super(_ReSkipNet, self).__init__()
        self.depth = depth
        assert self.depth % 6 == 2, 'Depth must be = 6n + 2!'
        self.name = 'ReSkipNet'
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        n = (depth - 2) // 6
        block, layers = (BasicBlock, [n]*3)
        self.num_layers = layers
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self._make_layer(block, 64, layers[0], stride=1, pool_size=32, group_id=1)
        self._make_layer(block, 128, layers[1], stride=2, pool_size=16, group_id=2)
        self._make_layer(block, 256, layers[2], stride=2, pool_size=8, group_id=3, last_gate=False)
        self.rl_mode = rl_mode
        if self.rl_mode:
            self.control = RNNGatePolicy(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        self.fc = nn.Linear(256, num_classes)
        self._initialize_weights()
        self.inference_mode = inference_mode
        self.flop_weights = []
        self.nb_gates = self.get_nb_gates()

    def forward(self, x):

        if self.inference_mode:
            batch_size = x.size(0)
            x = self.relu(self.bn1(self.conv1(x)))
            self.control.hidden = self.control.init_hidden(batch_size)

            x = getattr(self, 'group1_layer0')(x)
            gate_feature = getattr(self, 'group1_gate0')(x)
            mask, gprob = self.control(gate_feature)

            if mask.sum().item() == 0:
                mask[0] = 1
            elif mask.sum().item() == batch_size:
                mask[0] = 0

            prev = x
            prev = prev[mask.squeeze() == 0]
            x = x[mask.squeeze() == 1]

            for g in range(3):
                for i in range(0 + int(g == 0), self.num_layers[g]):
                    if getattr(self, 'group{}_ds{}'.format(g + 1, i)) is not None:
                        prev = getattr(self, 'group{}_ds{}'.format(g + 1, i))(prev)
                    x = getattr(self, 'group{}_layer{}'.format(g + 1, i))(x)
                    res = torch.cuda.FloatTensor(batch_size, x.size(1), x.size(2), x.size(3))
                    res[mask.squeeze() == 0] = prev
                    res[mask.squeeze() == 1] = x
                    prev = x = res
                    if not (g == 2 and (i == self.num_layers[g] - 1)):
                        gate_feature = getattr(self, 'group{}_gate{}'.format(g + 1, i))(x)
                        mask, gprob = self.control(gate_feature)

                        if mask.sum().item() == 0:
                            mask[0] = 1
                        elif mask.sum().item() == batch_size:
                            mask[0] = 0

                        prev = prev[mask.squeeze() == 0]
                        x = x[mask.squeeze() == 1]

            x = nn.AvgPool2d(x.size()[3])(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

        else:
            batch_size = x.size(0)
            x = self.relu(self.bn1(self.conv1(x)))
            self.control.hidden = self.control.init_hidden(batch_size)

            masks = []
            gprobs = []
            x = getattr(self, 'group1_layer0')(x)
            gate_feature = getattr(self, 'group1_gate0')(x)
            mask, gprob = self.control(gate_feature)

            gprobs.append(gprob)
            masks.append(mask.squeeze())
            prev = x

            for g in range(3):
                for i in range(0 + int(g == 0), self.num_layers[g]):
                    if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                        prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                    input_res = x.shape[1:]
                    x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                    if len(self.flop_weights) < self.nb_gates:
                        module = getattr(self, 'group{}_layer{}'.format(g + 1, i))
                        flops = get_total_flops(module, input_res)
                        self.flop_weights.append(flops)
                    prev = x = mask.expand_as(x) * x + (1 - mask).expand_as(prev)*prev
                    if not (g == 2 and (i == self.num_layers[g] - 1)):
                        gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                        mask, gprob = self.control(gate_feature)
                        gprobs.append(gprob)
                        masks.append(mask.squeeze())

            x = nn.AvgPool2d(x.size()[3])(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, masks, gprobs

    def _make_layer(self, block, planes, blocks, stride=1, pool_size=32, group_id=1, last_gate=True):
        layer = block(inplanes=self.inplanes, planes=planes, stride=stride)
        self.inplanes = planes*block.expansion
        gate_layer = GateLayer(embed_dim=self.embed_dim, pool_size=pool_size, inplanes=self.inplanes)
        setattr(self, 'group{}_ds{}'.format(group_id, 0), layer.downsample)
        setattr(self, 'group{}_layer{}'.format(group_id, 0), layer)
        setattr(self, 'group{}_gate{}'.format(group_id, 0), gate_layer)
        for k in range(1, blocks):
            layer = block(inplanes=self.inplanes, planes=planes)
            gate_layer = GateLayer(embed_dim=self.embed_dim, pool_size=pool_size, inplanes=self.inplanes)
            setattr(self, 'group{}_layer{}'.format(group_id, k), layer)
            setattr(self, 'group{}_ds{}'.format(group_id, k), layer.downsample)
            skip_gate = (k == blocks - 1) and not last_gate
            if not skip_gate:
                setattr(self, 'group{}_gate{}'.format(group_id, k), gate_layer)

    def add_basis(self):
        self.conv1 = BasisLayer(self.conv1)
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                module.add_basis()
        self.fc = BasisLayer(self.fc)

    def get_additional_flops(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_flops()
        return res

    def get_additional_params(self):
        res = 0
        for module in self.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
               res += module.get_additional_parameters()
        return res

    def get_nb_gates(self):
        res = 0
        for module in self.modules():
            if isinstance(module, GateLayer):
                res += 1
        return res

    def fill_flop_weights(self):
        gate_saved_actions = self.control.saved_actions
        gate_rewards = self.control.rewards
        self.eval()
        input = torch.cuda.FloatTensor(2, 3, 32, 32)
        with torch.no_grad():
            out = self.forward(input)
        del gate_saved_actions[:]
        del gate_rewards[:]
        all_flops = sum(self.flop_weights)
        self.flop_weights = [flop/all_flops for flop in self.flop_weights]


def _reskipnet(num_classes, depth):
    kwargs = {'num_classes': num_classes, 'depth': depth}
    return _ReSkipNet(**kwargs)


def reskipnet(num_classes, depth):
    kwargs = {'num_classes': num_classes, 'depth': depth}
    assert depth % 6 == 2, 'Be carreful! Depth should satisfies: depth % 6 == 2'
    return _reskipnet(**kwargs)


