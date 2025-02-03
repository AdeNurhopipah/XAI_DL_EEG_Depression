"""
This module includes modified CNN models for EEG Depression classification tasks. 
1. TempCNN : Acharyaa, et al (2018), doi: 10.1016/j.cmpb.2018.04.012
2. TwoDCNN : Ravindran & Contreras‐Vidal (2023), doi: 10.1038/s41598-023-43871-8
3. EEGNet : Liu, et al (2022), doi: 10.3389/fpsyt.2022.864393
4. DeprNet : Seal et al. (2021), doi : 10.1109/TIM.2021.3053999
5. ResNet : Kang, et al. (2023), doi : 10.1109/TNSRE.2023.3293051

Note:
- All activation functions in the models use ReLU to standardise the XAI implementation across Captum and PyTorch Grad-CAM, 
ensuring consistency in gradient-based interpretability methods.
- Remove Softmax to keep only the raw logits, preserving the meaningful gradients necessary for effective feature attribution. 
- Same padding ensures consistent output size, retains more features, and improves gradient flow.
"""      
 
from torch import nn

class TempCNN(nn.Module):
    def __init__(self, n_class=4, n_samples=2000, n_chan=1):
        super(TempCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(1,5), stride=1, padding='same')
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(1,5), stride=1, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(1,5), stride=1, padding='same')
        self.relu3 = nn.ReLU() 
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,5), stride=1, padding='same')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(1,5), stride=1, padding='same')
        self.relu5 = nn.ReLU() 
        self.pool5 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*n_chan*(n_samples//32),80) 
        self.relu6 = nn.ReLU() 
        self.fc2 = nn.Linear(80, 40)
        self.relu7 = nn.ReLU() 
        self.fc3 = nn.Linear(40, n_class)
        

    def forward(self, x):
        # Convolution and pooling layers
        x = self.relu1(self.conv1(x))  
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x)) 
        x = self.pool3(x)
        x = self.relu4(self.conv4(x))
        x = self.pool4(x)
        x = self.relu5(self.conv5(x))  
        x = self.pool5(x)

        x = self.flatten(x)

        # Fully connected layers
        x = self.relu6(self.fc1(x)) 
        x = self.relu7(self.fc2(x))  
        x = self.fc3(x) 

        return x


class TwoDCNN(nn.Module):
    def __init__(self, n_class, n_chan=62, n_samples=250):
        super(TwoDCNN, self).__init__()

        #Convolutional, pooling and flatten layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,5), stride=1, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,5), stride=1, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,5), stride=1, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,5), stride=1, padding='same')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,5), stride=1, padding='same')
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(n_chan,1), stride=1, padding='valid')
        self.relu6 = nn.ReLU()
        self.flatten  = nn.Flatten()

        #Linear and dropout layers
        self.fc1  = nn.Linear(in_features= 32*(n_samples//16), out_features=32)
        self.dropout = nn.Dropout(0.5)
        self.fc2    = nn.Linear(in_features=32, out_features=n_class)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        x = self.relu4(self.conv4(x))
        x = self.pool4(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, n_class=4, n_chan=128, n_samples=125, dropoutRate=0.5, kernLength=100, F1=8, D=2, F2=16): 
        super(EEGNet, self).__init__()

        # First convolutional layer 
        self.conv2d1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolutional layer
        self.dwc = nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False, padding='valid')
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.relu1 = nn.ReLU()
        self.ap1 = nn.AvgPool2d((1, 4))
        self.do1 = nn.Dropout(dropoutRate)

        # Separable convolutional layer
        self.sc = nn.Conv2d(F1 * D, F2, (1, 16), bias=False, padding='same')
        self.bn3 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU()
        self.ap2 = nn.AvgPool2d((1, 8))
        self.do2 = nn.Dropout(dropoutRate)
        
        #Dense layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * ((n_samples // 4) // 8), n_class) 
        
    def forward(self, x):
        x = self.conv2d1(x)
        x = self.bn1(x) 
        x = self.dwc(x)
        x = self.bn2(x)
        x = self.relu1(x)
        x = self.ap1(x)
        x = self.do1(x)
        x = self.sc(x)
        x = self.bn3(x)
        x = self.relu2(x)
        x = self.ap2(x)
        x = self.do2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return  x
    

class DeprNet(nn.Module):
    def __init__(self,  n_class=4, n_chan=5, n_samples=2000):
        super(DeprNet, self).__init__()

        # Convolutional layers
        #original stridde = (1,2) adjusted for the 'same' padding
        self.conv1 = nn.Conv2d(1, 128, (1, 5), stride=(1, 1), padding= 'same') 
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(128, 64, (1, 5), stride=(1, 1), padding= 'same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(64, 64, (1, 5), stride=(1, 1), padding= 'same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.conv4 = nn.Conv2d(64, 32, (1, 3), stride=(1, 1), padding= 'same')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(32, 32, (1, 2), stride=(1, 1), padding= 'same')
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((1, 2), stride=(1, 2))

        # Fully connected and flatten layers
        self.fc1 = nn.Linear(32*n_chan*(n_samples//32), 16) 
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, n_class)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn2(self.conv2(x))
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn3(self.conv3(x))
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn4(self.conv4(x))
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.bn5(self.conv5(x))
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
     
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Shortcut connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) \
            if in_channels != out_channels or stride != 1 else nn.Identity()
        
    def forward(self, x):
        identity =  self.skip(x) 
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
                         
        # Ensure spatial dimensions match
        if identity.size() != out.size():
            identity = nn.functional.adaptive_avg_pool2d(identity, out.shape[2:])

        out = out+ identity
        return out


class ResNet(nn.Module):

    def __init__(self, n_class=2, n_chan=19, n_samples=64):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.n_chan = n_chan
        self.n_samples = n_samples
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # Residual blocks (manual stacking)
        self.layer1_block = ResidualBlock(1, 64, stride=1)
        self.layer2_block = ResidualBlock(64, 128, stride=2)
        self.layer3_block = ResidualBlock(128, 256, stride=2)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, n_class)
        self.flatten = nn.Flatten()

    def forward(self, x):
        #Original model repeted 4 Residual block
        x = self.relu1(self.layer1_block(x))
        x = self.relu2(self.layer2_block(x))
        x = self.relu3(self.layer3_block(x))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def SelectModel(model_name, n_class=4, n_chan=62, n_samples =250):
    if model_name == 'TempCNN':
        model_sel = TempCNN(n_class=n_class, n_chan=n_chan, n_samples=n_samples)
    elif model_name == 'TwoDCNN':
        model_sel = TwoDCNN(n_class=n_class, n_chan=n_chan, n_samples=n_samples)
    elif model_name == 'EEGNet':
        model_sel = EEGNet(n_class=n_class, n_chan=n_chan, n_samples=n_samples)
    elif model_name == 'DeprNet':
        model_sel = DeprNet(n_class=n_class, n_chan=n_chan, n_samples=n_samples)
    elif model_name == 'ResNet':
        model_sel = ResNet(n_class=n_class, n_chan=n_chan, n_samples=n_samples)
    else:
        print('Invalid model name')
        return None
    
    return model_sel