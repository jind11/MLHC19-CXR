import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
from tifffile import imread
import torchvision.models as models
import os
import sys
from sklearn.metrics import roc_auc_score
import numpy as np
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

'''
Embedding Model Paths
'''
joint32_model = "exp/lr1.000000e-03-epochs50-bs32-joint-space32-fixeddata-w10/model.best.pth"
joint64_model = "exp/lr1.000000e-03-epochs50-bs32-joint-space64-fixeddata-w10/model.best.pth"
logreg32_model = "exp/lr1.000000e-03-epochs50-bs32-logreg-space32-fixeddata/model.best.pth"
logreg64_model = "exp/lr1.000000e-03-epochs50-bs32-logreg-space64-fixeddata/model.best.pth"
recall32_model = "exp/lr1.000000e-03-epochs50-bs32-recall-space32-fixeddata/model.best.pth"
recall64_model = "exp/lr1.000000e-03-epochs50-bs32-recall-space64-fixeddata/model.best.pth"

emb_dims = 32

'''
Embedding Model Classes
'''
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

class Model_JointLearning(nn.Module):
  def __init__(self):
    super(Model_JointLearning, self).__init__()
    self.image_branch = ModelImages()
    self.text_branch = ModelReports()
    self.classification_branch = ModelClassification()
  def forward(self, images, text):
    image_emb = self.image_branch(images)
    text_emb = self.text_branch(text)
    predictions = self.classification_branch(image_emb)
    return image_emb, text_emb, predictions

class Model_RecallLogreg(nn.Module):
  def __init__(self):
    super(Model_RecallLogreg, self).__init__()
    self.image_branch = ModelImages()
    self.text_branch = ModelReports()
  def forward(self, images, text):
    image_emb = self.image_branch(images)
    text_emb = self.text_branch(text)
    return image_emb, text_emb

'''
Classification Model
'''
class Model_Classification(nn.Module):

  def __init__(self):
    super(Model_Classification, self).__init__()

    global emb_dims
    self.densenet = DenseNetTruncated(pretrained=True)
    self.last_layer = nn.Linear(1024 + emb_dims, 14)

  def forward(self, image, image_emb):
    x = self.densenet(image)
    x = torch.cat((x, image_emb), dim=1)
    output = self.last_layer(x)
    return output

'''
Dataset
'''
class MIMICCXRDataset(Dataset):
  def __init__(self, train=False, valid_train=False):

    self.train = train
    self.valid_train = valid_train
    # Load all of the text into memory
    # Pre-compute TF-IDF or whatever other features as a sparse matrix
  
    if train:
      images_path = 'data/images_train'
      labels_path = 'data/labels_train.npy'
    else:
      images_path = 'data/images_test'
      labels_path = 'data/labels_test.npy'
    
    images_list = os.listdir(images_path)
    self.images = [
      {
        "exid": int(filename.split("_")[0]),
        "file": os.path.join(images_path, filename)
      }
      for filename in sorted(images_list, key=lambda x: int(x.split("_")[0]))]

    self.labels = np.load(labels_path, allow_pickle=True, encoding='bytes')

    #print(self.labels)
    
    self.indices = set([img['exid'] for img in self.images])
    self.images_map = {}
    for i, image in enumerate(self.images):
      if image['exid'] in self.indices:
        self.images_map[image['exid']] = i

    self.indices = sorted(self.indices)
    self.test_cutoff_index = len(self.indices) // 2

  def __len__(self):
    if self.train == True:
      return len(self.indices)
    if self.valid_train == True:
      return self.test_cutoff_index
    return len(self.indices) - self.test_cutoff_index

  def __getitem__(self, index):
    if self.train:
      new_index = 0 + index
    else:
      if self.valid_train:
        new_index = 0 + index
      else:
        new_index = self.test_cutoff_index + index

    image = self.images[self.images_map[self.indices[new_index]]]

    image_orig = imread(image['file'])
    image_orig = np.expand_dims(image_orig, axis=0)
    image = np.concatenate((image_orig, image_orig, image_orig), axis=0)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_norm = normalize(torch.from_numpy(np.divide(image, np.max(image)).astype(np.float32)))

    if not self.train:
      labels = torch.from_numpy(self.labels[self.indices[new_index],:].astype(np.float32))
    else:
      labels = torch.from_numpy(self.labels[self.indices[new_index],:].astype(np.float32))

    return image_norm, labels

'''
Measurement Functions
'''
class EMAMeter:
  def __init__(self, decay=0.95):
    self.decay = decay
    self.value = None

  def update(self, value):
    if self.value is None:
      self.value = value
    else:
      self.value = self.value * self.decay + value * (1 - self.decay)

def get_loss(predictions, labels, gpu):
  if gpu:
    predictions = predictions.cuda()
    labels = labels.cuda()

  predictions = torch.reshape(predictions, (-1,))
  labels = torch.reshape(labels, (-1,))

  indices_to_use = (labels > -.5).nonzero()
  loss = nn.BCEWithLogitsLoss()
  output = loss(predictions[indices_to_use], labels[indices_to_use])
  return output

def get_auc(predictions, labels, gpu):
  predictions_np = predictions.detach().cpu().numpy()
  labels_np = labels.detach().cpu().numpy()

  auc_per_class = [None for _ in range(14)]

  for label in range(14):
    predictions_per_label = predictions_np[:,label]
    labels_per_label = labels_np[:,label]

    indices_to_use = (labels_per_label > -.5).nonzero()
    predictions_per_label = predictions_per_label[indices_to_use]
    labels_per_label = labels_per_label[indices_to_use]
    labels_per_label = (labels_per_label > .5).astype(np.int32)
    #print(labels_per_label)
    if not (max(labels_per_label) == min(labels_per_label)):
      auc = roc_auc_score(labels_per_label, predictions_per_label)
      auc_per_class[label] = auc

  return auc_per_class

'''
Training the Model
'''
def validate_model(model, dataloader_validation, gpu, embedding_model):
  print("Validating model...")
  global handle
  res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
  #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
  if gpu:
    embedding_model = embedding_model.cuda()

  predictions_list = []
  labels_list = []
  for i, (images, labels) in enumerate(dataloader_validation):
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    #print(labels)
    labels_list.append(labels)
    if gpu:
      images = images.cuda()
    images_emb = embedding_model.image_branch(images)
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    predictions = model(images, images_emb)
    predictions_list.append(torch.sigmoid(predictions.cpu().detach()))
    

  all_predictions = torch.cat(predictions_list)
  all_labels = torch.cat(labels_list)

  auc = get_auc(all_predictions, all_labels, gpu)

  return auc

def test_model_save(model, dataloader_test, gpu, embedding_model):
  if gpu:
    embedding_model = embedding_model.cuda()

  labels_list = []
  embeddings = []
  for i, (images, labels) in enumerate(dataloader_test):
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    #print(labels)
    labels_list.append(labels)
    if gpu:
      images = images.cuda()
    images_emb = embedding_model.image_branch(images)
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    embeddings.append(images_emb.cpu().detach())

  all_emb = torch.cat(embeddings).numpy()
  all_labels = torch.cat(labels_list).numpy()

  np.save("labels_save_32.npy", all_labels)
  np.save("emb_save_32.npy", all_emb)


def test_model(model, dataloader_test, gpu, embedding_model):
  print("Testing model...")
  if gpu:
    embedding_model = embedding_model.cuda()
    model = model.cuda()

  predictions_list = []
  labels_list = []
  predictions_embedding_list = []
  for i, (images, labels) in enumerate(dataloader_test):
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    #print(labels)
    labels_list.append(labels)
    if gpu:
      images = images.cuda()
    images_emb = embedding_model.image_branch(images)
    #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    predictions = model(images, images_emb)
    predictions_list.append(torch.sigmoid(predictions.cpu().detach()))
    predictions_emb = embedding_model.classification_branch(images_emb)
    predictions_embedding_list.append(torch.sigmoid(predictions_emb.cpu().detach()))
    

  all_predictions = torch.cat(predictions_list)
  all_labels = torch.cat(labels_list)
  all_predictions_emb = torch.cat(predictions_embedding_list)

  auc = get_auc(all_predictions, all_labels, gpu)
  print("Test AUC for Classification Task:")
  print(auc)

  auc = get_auc(all_predictions_emb, all_labels, gpu)
  print("Test AUC for Embedding Network Classification Task:")
  print(auc)

def train_epoch(epoch, optim, model, dataloader_train, gpu, embedding_model):
  if gpu:
    embedding_model = embedding_model.cuda()

  model.train()
  loss_meter = EMAMeter(0.6)
  for i, (images, labels) in enumerate(dataloader_train):
    if gpu:
      images = images.cuda()

    images_emb = embedding_model.image_branch(images)
    optim.zero_grad()
    predictions = model(images, images_emb)

    loss = get_loss(predictions, labels, gpu)
    loss.backward()
    optim.step()

    loss = float(loss.cpu())
    loss_meter.update(loss)
    if i % 25 == 10:
      print("Epoch %d. Step %d / %d. Loss %.3f" % (epoch, i, len(dataloader_train), loss_meter.value))
      #break
    global handle
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    #print(f'EACH TRAINING BATCH gpu: {res.gpu}%, gpu-mem: {res.memory}%')

def train_model(dataloader_train, dataloader_validation, gpu, model_emb, model_type, epochs, lr):
  # Create experiment directory
  global emb_dims
  expdir = "exp_classification/" + model_type + "-" + str(emb_dims)
  os.makedirs(expdir, exist_ok=True)

  # Create your model
  model = Model_Classification()
  if gpu:
    model = model.cuda()

  # Create your optimizer
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  model.train()

  best_valid_auc = 0
  for epoch in range(epochs):
    train_epoch(epoch, optim, model, dataloader_train, gpu, model_emb)
    global handle
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    #print(f'BEFORE VALID gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    valid_auc = validate_model(model, dataloader_validation, gpu, model_emb)
    avg_auc = sum(x if x != None else 0 for x in valid_auc) / sum(x is not None for x in valid_auc)
    print("Epoch Validation AUC all labels: ", valid_auc, " Avg AUC: ", avg_auc)

    if avg_auc > best_valid_auc:
      save_model(expdir, epoch, optim, model, valid_auc, best_valid_auc, is_best=True)
      best_valid_auc = avg_auc
    else:
      save_model(expdir, epoch, optim, model, valid_auc, best_valid_auc, is_best=False)

'''
Saving the Model
'''
def save_model(expdir, epoch, optim, model, valid_auc, best_valid_auc, is_best=False):
  ckpt = os.path.join(expdir, "model.pth")
  best = os.path.join(expdir, "model.best.pth")
  torch.save({
    "model": model.state_dict(),
    "optim": optim.state_dict(),
    "epoch": epoch,
    "auc": valid_auc,
    "best_auc": best_valid_auc,
  }, ckpt)
  if is_best:
    if os.path.exists(best):
      os.unlink(best)
    os.link(ckpt, best)


def main():
  global joint32_model, joint64_model, recall32_model, recall64_model, logreg32_model, logreg64_model
  global emb_dims

  # ARGUMENTS: embedding_model emb_dims gpu
  emb_model = sys.argv[1]
  emb_dims = int(sys.argv[2])
  gpu = True if sys.argv[3] == "True" else False
  model = None

  if emb_model == "joint":
    if emb_dims == 32:
      model_state_dict = torch.load(joint32_model)["model"]
      model = Model_JointLearning()
      model.load_state_dict(model_state_dict)
    else:
      model_state_dict = torch.load(joint64_model)["model"]
      model = Model_JointLearning()
      model.load_state_dict(model_state_dict)
  elif emb_model == "recall":
    if emb_dims == 32:
      model_state_dict = torch.load(recall32_model)["model"]
      model = Model_RecallLogreg()
      model.load_state_dict(model_state_dict)
    else:
      model_state_dict = torch.load(recall32_model)["model"]
      model = Model_RecallLogreg()
      model.load_state_dict(model_state_dict)
  elif emb_model == "logreg":
    if emb_dims == 32:
      model_state_dict = torch.load(logreg32_model)["model"]
      model = Model_RecallLogreg()
      model.load_state_dict(model_state_dict)
    else:
      model_state_dict = torch.load(logreg64_model)["model"]
      model = Model_RecallLogreg()
      model.load_state_dict(model_state_dict)

  dataset_train = MIMICCXRDataset(train=True)
  dataset_validation = MIMICCXRDataset(train=False, valid_train=False)
  dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
  dataloader_validation = DataLoader(dataset_validation, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

  train_model(dataloader_train, dataloader_validation, gpu, model, emb_model, 50, .001)

def get_test_results():
  dataset_test = MIMICCXRDataset(train=False, valid_train=True)
  dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

  global joint32_model, joint64_model, recall32_model, recall64_model, logreg32_model, logreg64_model
  global emb_dims

  # ARGUMENTS: embedding_model emb_dims gpu
  emb_dims = 64
  gpu = True

  model_state_dict = torch.load(joint64_model)["model"]
  model = Model_JointLearning()
  model.load_state_dict(model_state_dict)

  model_classify_state_dict = torch.load("exp_classification/joint-64/model.best.pth")["model"]
  model_classify = Model_Classification()
  model_classify.load_state_dict(model_classify_state_dict)

  test_model(model_classify, dataloader_test, gpu, model)

def generate_embeddings_save():
  dataset_test = MIMICCXRDataset(train=False, valid_train=False)
  dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

  global joint32_model, joint64_model, recall32_model, recall64_model, logreg32_model, logreg64_model
  global emb_dims

  # ARGUMENTS: embedding_model emb_dims gpu
  emb_dims = 32
  gpu = True

  model_state_dict = torch.load(joint32_model)["model"]
  model = Model_JointLearning()
  model.load_state_dict(model_state_dict)

  model_classify_state_dict = torch.load("exp_classification/joint-32/model.best.pth")["model"]
  model_classify = Model_Classification()
  model_classify.load_state_dict(model_classify_state_dict)

  test_model_save(model_classify, dataloader_test, gpu, model)


if __name__ == '__main__':
  main()
