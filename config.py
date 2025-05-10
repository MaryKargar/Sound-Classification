# === config.py (Optimized for CPU training + SpecAugment) ===
import os

esc50_path = 'data/esc50/esc50.csv'
audio_dir = 'data/esc50/audio'
runs_path = 'results'
disable_bat_pbar = False

n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]

sr = 44100
n_mels = 128
hop_length = 512

dropout_p = 0.1
use_attention = True
model_constructor = "AudioMLP(num_classes=config.n_classes, dropout_p=config.dropout_p, use_attention=config.use_attention)"

val_size = 0.2
device_id = 0
batch_size = 8
num_workers = 0
persistent_workers = False
epochs = 100
patience = 10

lr = 1e-3
weight_decay = 1e-3
warm_epochs = 5
gamma = 0.8
step_size = 5

test_checkpoints = ['terminal.pt']

test_experiment = max(
    [os.path.join(runs_path, d) for d in os.listdir(runs_path)
     if os.path.isdir(os.path.join(runs_path, d)) and d != 'sample-run'],
    key=os.path.getmtime
)
