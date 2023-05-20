import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import flask
from flask import request

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = functional.cross_entropy(out, labels)
        acc = get_accuracy(out, labels)
        return {'valid_loss': loss.detach(), 'valid_acc': acc}

    def validation_end(self, outputs):
        batch_losses = [x['valid_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['valid_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'valid_loss': epoch_loss.item(), 'valid_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print('{} epoch: train_loss = {:.4f}, valid_loss = {:.4f}, valid_acc = {:.4f}'
              .format(epoch + 1, result['train_loss'], result['valid_loss'], result['valid_acc']))

class ResNet50(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained = True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def get_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def predict_for_internal_image(image):
    x_image = image.unsqueeze(0)
    y_image = model(x_image)
    probability, class_number = torch.max(y_image, dim = 1)
    class_name = classes[class_number[0].item()]
    return class_name, class_number[0].item(), probability[0].item()

def predict_for_external_image(image_name):
    image = Image.open(Path('./' + image_name))
    image = data_transform(image)
    class_name, class_number, probability = predict_for_internal_image(image)
    return class_name, class_number, probability

data_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
classes = ['glass', 'metal', 'paper', 'plastic', 'trash']
model_name = 'ResNet50_trained_model.pt'
model = torch.load(Path(model_name), map_location = torch.device('cpu'))
model.eval()

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def connection_request():
    return "Successful Connection"

@app.route('/predictClass', methods=['GET', 'POST'])
def prediction_request():
    if request.method == 'POST':
        file = request.files['external_image']
        image_name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
        class_name, class_number, probability = predict_for_external_image(image_name)
        print('There is photo of {} ({} class, probability = {:.4f})'.format(class_name, class_number, probability))
        return class_name
    else:
        return ""

app.config['UPLOAD_FOLDER'] = ""
app.run(host = "0.0.0.0", port = 5000, debug = False)