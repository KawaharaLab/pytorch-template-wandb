import torch
import glob
import os
import csv
from torchvision import datasets, transforms
import torchaudio
from base import BaseDataLoader
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
class TongueDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, frame_width, sample_rate):
        super().__init__()
        self.data = []
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate) # 48 kHz
        self.gesture_dict = {"tap": 0, "click": 1}
        self.direction_dict = {"left": 0, "middle": 1, "right": 2}
        self.resizer = transforms.Resize((224, 224))
        self.load_data(data_dir, frame_width, sample_rate)
    def load_data(self, data_dir, frame_width, sample_rate):
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
        for csv_file in csv_files:
            wav_file = csv_file.replace(".csv", ".wav")
            wav, fs = torchaudio.load(wav_file)
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    num = int(row['num'])
                    gesture = self.gesture_dict[row['gesture']]
                    direction = self.direction_dict[row['direction']]
                    start_idx = num - frame_width // 2
                    end_idx = start_idx + frame_width
                    wav_segment = wav[:, start_idx:end_idx]
                    mel_spec = self.transform(wav_segment)
                    mel_spec_resized = self.resizer(mel_spec)
                    # make data from 2 channel to 3 channel
                    mel_spec_resized = mel_spec_resized[(0, 0, 1), :, :]
                    self.data.append((mel_spec_resized, gesture, direction))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
class TongueDataLoader(BaseDataLoader):
    """
    Tongue data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, frame_width=4096, sample_rate=48000):
        self.data_dir = data_dir
        self.dataset = TongueDataset(self.data_dir, frame_width, sample_rate)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers) 