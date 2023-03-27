import pandas as pd
import os
import math, random
import torchaudio
import matplotlib.pyplot as plt
from torchaudio import transforms
from IPython.display import Audio
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from torch.utils.data import ConcatDataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import torch; print(torch.version.cuda); print(torch.cuda.is_available())
import pytorch_lightning as pl 
import numpy as np
import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Load an audio file. Return the signal as a tensor and the sample rate
# ----------------------------
class AudioUtil():
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
  
  # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  # ----------------------------
  # Since Resample applies to a single channel, we resample one channel at a time
  # ----------------------------
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))
    
  # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)
  
  # ----------------------------
  # Shifts the signal to the left or right by some percent. Values at the end
  # are 'wrapped around' to the start of the transformed signal.
  # ----------------------------
  @staticmethod
  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

from torch.utils.data import DataLoader, Dataset

data_path = os.getcwd()

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 4000
    self.sr = 44100
    self.channel = 2
    self.shift_pct = 0.4
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # Get the Class ID
    poa = self.df.loc[idx, 'poa']

    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)

    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)


    return sgram, poa

import pytorch_lightning as pl 

#For Metrics
class Metric:
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []

    def update(self, y, t):
        '''Update with batch outputs and labels.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        '''
        self.y.append(y)
        self.t.append(t)

    def _process(self, y, t):
        '''Compute TP, FP, FN, TN.
        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        tp = torch.empty(self.num_classes)
        fp = torch.empty(self.num_classes)
        fn = torch.empty(self.num_classes)
        tn = torch.empty(self.num_classes)
        for i in range(self.num_classes):
            tp[i] = ((y == i) & (t == i)).sum().item()
            fp[i] = ((y == i) & (t != i)).sum().item()
            fn[i] = ((y != i) & (t == i)).sum().item()
            tn[i] = ((y != i) & (t != i)).sum().item()
        return tp, fp, fn, tn

    def accuracy(self, reduction='mean'):
        '''Accuracy = (TP+TN) / (P+N).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) accuracy.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        if reduction == 'none':
            acc = tp / (tp + fn)
        else:
            acc = tp.sum() / (tp + fn).sum()
        return acc

    def precision(self, reduction='mean'):
        '''Precision = TP / (TP+FP).
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) precision.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'mean':
            prec = prec.mean()
        return prec

    def recall(self, reduction='mean'):
        '''Recall = TP / P.
        Args:
          reduction: (str) mean or none.
        Returns:
          (tensor) recall.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'mean':
            recall = recall.mean()
        return recall

    def confusion_matrix(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((y == i) & (t == j)).sum().item()
        return matrix


def test():
    import torchmetrics.functional as M
    nc = 2
    m = Metric(num_classes=nc)
    y = torch.randint(0, nc, (2,))
    t = torch.randint(0, nc, (2,))
    m.update(y, t)

    print('\naccuracy:')
    print(m.accuracy('none'))
    print(M.accuracy(y, t, 'none', num_classes=2))
    print(m.accuracy('mean'))
    

    print('\nprecision:')
    print(m.precision('none'))
    print(M.precision(y, t, 'none', num_classes=2))
    print(m.precision('mean'))
   

    print('\nrecall:')
    print(m.recall('none'))
    print(M.recall(y, t, 'none', num_classes=2))
    print(m.recall('mean'))
    

    print('\nconfusion matrix:')
    print(m.confusion_matrix())
    print(M.confusion_matrix(y, t, num_classes=2))


if __name__ == '__main__':
    test()

# Load model
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]
        
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
         
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x

model = AudioClassifier()
model.load_state_dict(torch.load(r"model.pt"))
model.eval()

import pandas as pd
import os

# Read metadata file
os.getcwd()
df = pd.read_csv('poachertest.csv')
df.head()

# Construct file path by concatenating datasetid and itemid
df['relative_path'] = '/' + df['itemid'].astype(str)

# Take relevant columns
df = df[['relative_path', 'poa']]

memeds = SoundDS(df, data_path)
# Create test data loaders
inf = torch.utils.data.DataLoader(memeds, batch_size=1, shuffle=False)
print(inf)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
model = model.to(device)

def inference (model, inf):
  correct_prediction = 0
  total_prediction = 0
  metric = Metric(num_classes=2)

  # Disable gradient updates
  with torch.no_grad():
    for images, labels in inf:

      # Get the predicted class with the highest score
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        _, predictions = torch.max(output.data,1)
        metric.update(predictions, labels)
        correct_prediction += (predictions == labels).sum().item()
        total_prediction += predictions.shape[0]
        
     
    # Metrics
    acc = metric.accuracy()*100
    prec = metric.precision()
    recall = metric.recall()
    f1 = 2*prec*recall / (prec+recall)
    confusion_matrix = metric.confusion_matrix()
    print(f'Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f} | Total items: {total_prediction}')
    print(confusion_matrix)

    # Plot confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np 

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

    ax.set_title('CNN1 Model Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    # Display the visualization of the Confusion Matrix.
    plt.show()

# Run inference on trained model with the validation set
inference(model, inf)


