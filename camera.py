import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image

class ResNetMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiLabel, self).__init__()
        self.name = "resnetclassifier"
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        # Attaching our classifier to the pretrained model
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet(x)
        x = self.sigmoid(x)
        return x

class ResNet(nn.Module):
    def init(self,num_classes):
        super(ResNet, self).init()
        self.name = "resnetclassifier"

        # Load the pre-trained ResNet model
        resnet = models.resnet152(pretrained=True)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(num_features,num_classes),
            nn.Softmax(dim=1)  #single label 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#####################################

# GPU or CPU
use_cuda = True
resnetclassifier = ResNet(20)

if use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    resnetclassifier.cuda()
    print('CUDA is available! Running on GPU...')
else:
    device = torch.device("cpu")
    print('CUDA is not available. Running on CPU...')

#####################################

# Load in the model checkpoint
state = torch.load('model_densenetclassifier_bs150_lr0.001_dr0.9_thresh0.5_epoch9', map_location=torch.device('cpu'))
state = {k.replace('module.', ''): v for k, v in state.items()}
resnetclassifier.load_state_dict(state)

resnetclassifier.eval()

#####################################

# Load in a single image to perform the prediction on
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Change the image path here
image = Image.open("/content/marisol-benitez-QvkAQTNj4zk-unsplash.jpg").convert('RGB')

input_img = transform(image).unsqueeze(0)

#####################################

# Define the 38 possible classes

classes: ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Set prediction threshold
threshold = 0.5

# Make the prediction
output = resnetclassifier(input_img)
output = output.detach().cpu()

best = (output >= threshold).int()

# Interpret the tensor and save the predicted classes
predicted_indices = best.squeeze().nonzero().flatten().tolist()
predicted_classes = [classes[i] for i in predicted_indices]

print(predicted_class)