import torchvision.models as models
import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', num_classes=64, pretrained=True)
        # for params in self.model.parameters():
        #     params.requires_grad = False
        self.preFinal = nn.Linear(64, 32)
        self.outlayer = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x): 
        return self.outlayer(self.relu(self.preFinal(self.model(x))))

    def get_activations_gradient(self):
        return self.gradients

    def forward_hook(self, module, input, output):
        self.gradients = torch.autograd.grad(output, input[0], grad_outputs=torch.ones_like(output))[0]



class DomainAdaptiveNet(nn.Module):
    def __init__(self):
        super(DomainAdaptiveNet, self).__init__()
        self.effnet = EfficientNet()
        # self.effnet.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_6_10.pth'))
        self.shared_layers = self.effnet.model
        self.classificationLayer = nn.Sequential(self.effnet.preFinal, self.effnet.relu, self.effnet.outlayer)
        self.domainClassificationLayer = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 2))

    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        diseaseClass = self.classificationLayer(shared_features)
        domainClass = self.domainClassificationLayer(shared_features)

        return shared_features, diseaseClass, domainClass



# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# input = torch.rand(4, 3, 224, 224).to(device)
# model = DomainAdaptiveNet().to(device)
# sharedFeature, diseaseClass, domainClass = model(input)
# print("Done")
