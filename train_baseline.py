# train_baseline.py
import os
import glob # For finding .h5 files
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import UNetSegmenterLitModel # Assuming model.py is in the same directory
from dataset import BraTSDataModule   # Assuming dataset.py is in the same directory

if __name__ == '__main__':
    pl.seed_everything(42, workers=True) # For reproducibility

    NUM_CLASSES = 3        # Number of output segmentation classes (e.g., ET, ED, NCR for BraTS)
    NUM_INPUT_CHANNELS = 4 # Number of input MRI modalities for BraTS
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4         # Adjust based on your GPU memory
    NUM_WORKERS = 2        # For DataLoader, set to 0 if you have issues on Windows
    MAX_EPOCHS = 50        # Number of epochs for baseline training
    TRAIN_VAL_SPLIT_RATIO = 0.8 # For splitting the .h5 files

    H5_DATA_DIR = "BraTS2020_training_data/content/data" # Use the path you confirmed

    print(f"Looking for .h5 files in: {os.path.abspath(H5_DATA_DIR)}")

    all_h5_files = sorted(glob.glob(os.path.join(H5_DATA_DIR, "volume_*_slice_*.h5")))

    if not all_h5_files:
        raise FileNotFoundError(f"No .h5 files found in {H5_DATA_DIR}. Please check the path and that files exist.")
    
    print(f"Found {len(all_h5_files)} .h5 files.")
    # --- End of Path Setup ---

    # Instantiate DataModule
    brats_datamodule = BraTSDataModule(
        all_h5_file_paths=all_h5_files,
        train_val_split_ratio=TRAIN_VAL_SPLIT_RATIO,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_input_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        seed=42 # Use the same seed for reproducible splits
    )

    # Instantiate Model
    unet_model = UNetSegmenterLitModel(
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )

    # Logger
    tensorboard_logger = TensorBoardLogger("tb_logs_baseline_h5", name="brats_unet_local_h5")

    # Callbacks
    checkpoint_dir = "checkpoints_baseline_h5" # Define checkpoint directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="unet-brats-local-h5-{epoch:02d}-{val_dice:.4f}",
        save_top_k=1,
        monitor="val_dice", # Monitor Dice score on validation set
        mode="max",         # Save model with highest Dice score
        save_last=True      # Ensure last.ckpt is saved
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_dice",
        patience=10, # Stop if val_dice doesn't improve for 10 epochs
        verbose=True,
        mode="max"
    )

    # Trainer
    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, # Or specify number of GPUs e.g., [0, 1] for two GPUs
        # deterministic=True # For stricter reproducibility, might slow down training
    )

    # --- Logic to find checkpoint for resuming ---
    # Path to the 'last.ckpt' file within your defined checkpoint directory
    resume_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")

    if not os.path.exists(resume_checkpoint_path):
        print(f"WARNING: 'last.ckpt' not found in {checkpoint_dir}. Starting training from scratch.")
        resume_checkpoint_path = None # Trainer will start fresh if ckpt_path is None
    else:
        print(f"INFO: Found 'last.ckpt'. Attempting to resume training from: {resume_checkpoint_path}")
    # --- End of checkpoint logic ---

    # Start training
    print("Starting local baseline training with .h5 data...")
    trainer.fit(unet_model, datamodule=brats_datamodule, ckpt_path=resume_checkpoint_path) # Pass the path here

    print("Local baseline training finished.")
    # Access the best model path after training is fully complete
    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")
    elif resume_checkpoint_path and os.path.exists(resume_checkpoint_path) and not checkpoint_callback.best_model_path:
        print(f"Training resumed and finished. Last checkpoint was: {resume_checkpoint_path}. Check logs for best metric if different from last.")
    else:
        print("No best model path attribute found on checkpoint_callback or training did not save a new best model.")