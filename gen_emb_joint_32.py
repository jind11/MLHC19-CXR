import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision.transforms import Normalize
from tifffile import imread
import os
import random

emb_dims = 32

# IMAGES NETWORK
class DenseNetTruncated(nn.Module):
  def __init__(self, **kwargs):
    super(DenseNetTruncated, self).__init__()
    self.densenet = models.densenet121(**kwargs)


  def forward(self, x):
    features = self.densenet.features(x)
    out = F.relu(features, inplace=True)
    out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
    return out

class ModelImages(nn.Module):
  def __init__(self, pretrained=True):
    global emb_dims
    super(ModelImages, self).__init__()
    self.densenet = DenseNetTruncated(pretrained=pretrained)
    self.last_linear = nn.Linear(1024, emb_dims)

  def forward(self, x):
    x = self.densenet(x)
    x = self.last_linear(x)
    return x

class ModelReports(nn.Module):
  def __init__(self):
    global emb_dims
    super(ModelReports, self).__init__()
    self.first_layer = nn.Linear(8919, 64)
    self.middle_layer = nn.Linear(64, 64)
    self.last_layer = nn.Linear(64, emb_dims)

  def forward(self, x):
    x = F.relu(self.first_layer(x))
    x = F.relu(self.middle_layer(x))
    x = F.relu(self.last_layer(x))
    return x

class ModelClassification(nn.Module):
  def __init__(self):
    global emb_dims
    super(ModelClassification, self).__init__()
    self.first_layer = nn.Linear(emb_dims, 14)

  def forward(self, x):
    x = self.first_layer(x)
    return x

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.image_branch = ModelImages()
    self.text_branch = ModelReports()
    self.classification_branch = ModelClassification()

  def forward(self, images, text):
    image_emb = self.image_branch(images)
    text_emb = self.text_branch(text)
    predictions = self.classification_branch(image_emb)
    return image_emb, text_emb, predictions

def sampled_margin_ranking_loss(emb1, emb2):
  # emb1=images emb2=reports
  N = emb1.shape[0]
  idx = np.random.randint(N - 1, size=[N])
  idx[np.arange(N) == idx] = N - 1
  emb1_neg = emb1[idx]
  emb2_neg = emb2[idx]

  similarity = lambda x, y: (x * y).sum(dim=1)
 
  IR_gt = similarity(emb1, emb2)
  IR_impR = similarity(emb1, emb2_neg)
  IR_impI = similarity(emb1_neg, emb2)
  zeros = torch.zeros_like(IR_impI)
  loss = torch.max(zeros, 1 + IR_impR - IR_gt) + torch.max(zeros, 1 + IR_impI - IR_gt)
  loss = loss.mean(0)
  return loss
  
class EMAMeter:
  def __init__(self, decay=0.95):
    self.decay = decay
    self.value = None

  def update(self, value):
    if self.value is None:
      self.value = value
    else:
      self.value = self.value * self.decay + value * (1 - self.decay)

def classification_loss(predictions, labels, gpu):
  num_labels = labels.numel()
  if gpu:
    predictions = predictions.cuda()
    labels = labels.cuda()

  predictions = torch.reshape(predictions, (-1,))
  labels = torch.reshape(labels, (-1,))

  indices_to_use = (labels > -.5) == True
  loss = nn.BCEWithLogitsLoss()
  output = loss(predictions[indices_to_use], labels[indices_to_use])
  return output

def train_epoch(epoch, optim, model, train_loader, w, gpu):
  model.train()
  loss_meter = EMAMeter(0.6)
  for i, (images, text, labels) in enumerate(train_loader):
    if gpu:
      images = images.cuda()
      text = text.cuda()
    optim.zero_grad()
    image_emb, text_emb, predictions = model(images, text)

    loss_1 = sampled_margin_ranking_loss(image_emb, text_emb)
    loss_2 = classification_loss(predictions, labels, gpu)
    #print("loss values: ", loss_1, loss_2)
    loss = loss_1 * torch.tensor(w).expand_as(loss_1) + (loss_2 * torch.tensor(00).expand_as(loss_2))
    loss.backward()
    optim.step()

    loss = float(loss.cpu())
    loss_meter.update(loss)
    if i % 10 == 0:
      print("Epoch %d. Step %d / %d. Loss %.3f" % (epoch, i, len(train_loader), loss_meter.value))
      # NOTE: DEBUG:
      # break

def save_model(expdir, epoch, optim, model, validation_loss, best_validation_loss, w, is_best=False):
  ckpt = os.path.join(expdir, "model.pth")
  best = os.path.join(expdir, "model.best.pth")
  torch.save({
    "model": model.state_dict(),
    "optim": optim.state_dict(),
    "epoch": epoch,
    "loss": validation_loss,
    "best_loss": best_validation_loss,
  }, ckpt)
  if is_best:
    if os.path.exists(best):
      os.unlink(best)
    os.link(ckpt, best)

def read_file(path):
  with open(path, 'rb') as f:
    return f.read()

class MIMICCXRDataset(Dataset):
  def __init__(self, train=False, valid_train=False):
    self.train = train
    self.valid_train = valid_train
    # Load all of the text into memory
    # Pre-compute TF-IDF or whatever other features as a sparse matrix
  
    if train:
      reports_path = 'data/reports_train'
      images_path = 'data/images_train'
      labels_path = 'data/labels_train.npy'
    else:
      reports_path = 'data/reports_test'
      images_path = 'data/images_test'
      labels_path = 'data/labels_test.npy'
    
    reports_list = os.listdir(reports_path)
    self.reports = [
      {
        "exid": int(filename.split("_")[0]),
        "text": read_file(os.path.join(reports_path, filename)).decode("utf8")
      }
      for filename in sorted(reports_list, key=lambda x: int(x.split("_")[0]))]
    
    images_list = os.listdir(images_path)
    self.images = [
      {
        "exid": int(filename.split("_")[0]),
        "file": os.path.join(images_path, filename)
      }
      for filename in sorted(images_list, key=lambda x: int(x.split("_")[0]))]

    self.labels = np.load(labels_path, allow_pickle=True, encoding='bytes')
    
    image_ids = set([img['exid'] for img in self.images])
    report_ids = set([rep['exid'] for rep in self.reports])
    self.indices = image_ids.intersection(report_ids)
    print(len(self.indices))
    self.reports_map = {}
    self.images_map = {}
    for i, report in enumerate(self.reports):
      if report['exid'] in self.indices:
        self.reports_map[report['exid']] = i
    for i, image in enumerate(self.images):
      if image['exid'] in self.indices:
        self.images_map[image['exid']] = i

    self.indices = sorted(self.indices)

    if not train:
      random.seed(42)
      random.shuffle(self.indices)
      self.indices = self.indices[:4500]

    self.vectorizer = pickle.loads(read_file('vectorizer.pickle'))

  def __len__(self):
    if self.train == True:
      return len(self.indices)
    if self.valid_train == True:
      return 3500
    return 1000

  def __getitem__(self, index):
    # Load image_i into a 3 x 224 x 224 pytorch tensor (use torchvision transformations to crop)
    # Load the row of the sparse matrix of text vectors, convert to a dense vector, text_i as
    #   a torch tensor
    if self.train:
      new_index = 0 + index
    else:
      if self.valid_train:
        new_index = 0 + index
      else:
        new_index = 3500 + index

    report = self.reports[self.reports_map[self.indices[new_index]]]
    image = self.images[self.images_map[self.indices[new_index]]]

    report_vec = torch.from_numpy(self.vectorizer.transform([report['text']]).todense()[0].astype(np.float32))[0]
    image_orig = imread(image['file'])
    image_orig = np.expand_dims(image_orig, axis=0)
    image = np.concatenate((image_orig, image_orig, image_orig), axis=0)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_norm = normalize(torch.from_numpy(np.divide(image, np.max(image)).astype(np.float32)))

    if not self.train:
      labels = torch.from_numpy(self.labels[self.indices[new_index],:].astype(np.float32))
    else:
      # not used so it doesnt matter right now TODO: change if doing joint objective
      labels = torch.from_numpy(self.labels[0,:].astype(np.float32))

    return image_norm, report_vec, labels

def validate_model(model, valid_loader, gpu):
  predictions_list = []
  labels_list = []
  for i, (images, reports, labels) in enumerate(valid_loader):
      if gpu:
        images = images.cuda()
      image_emb = model.image_branch(images)
      predictions = model.classification_branch(image_emb)
      predictions_list.append(predictions.cpu().detach())
      labels_list.append(labels)

  predictions = torch.cat(predictions_list, dim=0)
  labels = torch.cat(labels_list, dim=0)

  return classification_loss(predictions, labels, gpu)

def train(train_loader, validation_loader, valid_train_loader, w=1, lr=0.001, epochs=25, batch_size=32, expdir="exp", gpu=False, n_cpu=8):
  # Create experiment directory
  global emb_dims
  expdir = os.path.join(expdir, "lr%e-epochs%d-bs%d-joint-space%d-fixeddata-w%d" % (lr, epochs, batch_size, emb_dims, w))
  os.makedirs(expdir, exist_ok=True)

  # Create your model
  model = Model()
  if gpu:
    model = model.cuda()

  # Create your optimizer
  optim = torch.optim.Adam(model.parameters(), lr=lr)
  #optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-7, momentum=0.9)

  model.image_branch.train()
  model.text_branch.train()
  model.classification_branch.train()

  best_validation_loss = 100000000
  for epoch in range(epochs):
    train_epoch(epoch, optim, model, train_loader, w, gpu)
    validation_loss = float(validate_model(model, validation_loader, gpu))
    print("Epoch Validation Classification Loss: ", validation_loss)
    if validation_loss < best_validation_loss:
      save_model(expdir, epoch, optim, model, validation_loss, best_validation_loss, w, is_best=True)
      best_validation_loss = validation_loss
    else:
      save_model(expdir, epoch, optim, model, validation_loss, best_validation_loss, w, is_best=False)

def main():
  # generate_embedding.py lr epochs batch_size gpu n_cpu w

  print("\nSTARTING TRAINING...")

  if sys.argv[4] == 'True':
    gpu = True
  else:
    gpu = False

  train_ds = MIMICCXRDataset(train=True)
  train_loader = DataLoader(train_ds, batch_size=int(sys.argv[3]), shuffle=True, num_workers=int(sys.argv[5]), pin_memory=gpu)
  valid_train_ds = MIMICCXRDataset(train=False, valid_train=True)
  valid_train_loader = DataLoader(valid_train_ds, batch_size=int(sys.argv[3]), shuffle=True, num_workers=int(sys.argv[5]), pin_memory=gpu)
  valid_ds = MIMICCXRDataset(train=False, valid_train=False)
  valid_loader = DataLoader(valid_ds, batch_size=int(sys.argv[3]), shuffle=True, num_workers=int(sys.argv[5]), pin_memory=gpu)

  train(train_loader, valid_loader, valid_train_loader, lr=float(sys.argv[1]), epochs=int(sys.argv[2]), batch_size=int(sys.argv[3]), gpu=gpu, w=int(sys.argv[6]))

if __name__ == '__main__':
  main()
















