import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Modified for CIFAR: 3x3 kernel, no pooling
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

    def get_params(self):
        """Returns parameters split into backbone and head for optimizer groups"""
        backbone_params = [p for n, p in self.named_parameters() if 'classifier' not in n]
        head_params = list(self.classifier.parameters())
        return backbone_params, head_params

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

class IncrementalResNet(nn.Module):
    """
    Wrapper to handle incremental classes.
    """
    def __init__(self, n_initial_classes=10):
        super().__init__()
        self.backbone = ResNet18(num_classes=n_initial_classes)
        self.n_classes = n_initial_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def expand_classifier(self, n_new_classes):
        old_head = self.backbone.classifier
        old_weights = old_head.weight.data
        old_bias = old_head.bias.data
        
        in_features = old_head.in_features
        new_total_classes = self.n_classes + n_new_classes
        
        new_head = nn.Linear(in_features, new_total_classes)
        
        # Copy old weights
        new_head.weight.data[:self.n_classes] = old_weights
        new_head.bias.data[:self.n_classes] = old_bias
        
        self.backbone.classifier = new_head
        self.n_classes = new_total_classes
