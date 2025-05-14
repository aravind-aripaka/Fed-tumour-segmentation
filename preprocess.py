# preprocess_brats.py
import os
import glob
import numpy as np
import nibabel as nib # For loading NIfTI files

# --- Configuration ---
RAW_DATA_DIR = "BraTS2020_training_data" 
OUTPUT_DIR = "preprocessed_brats_2d"  
IMG_SIZE = (240, 240)                   
NUM_CLASSES = 3                        

TRAIN_VAL_SPLIT_RATIO = 0.8             
MIN_TUMOR_PIXELS_THRESHOLD = 100        

# --- End Configuration ---

def normalize_volume(volume):
    """
    Basic Z-score normalization for non-zero voxels of a 3D volume.
    Clips to 3 standard deviations and scales to [0, 1].
    This is a simple example; more sophisticated normalization might be needed.
    """
    mask = volume > np.min(volume) # Consider non-background/non-air voxels
    if np.sum(mask) == 0: # Avoid division by zero if volume is all one value (e.g. all zeros)
        return volume.astype(np.float32)

    mean = np.mean(volume[mask])
    std = np.std(volume[mask])
    
    if std < 1e-5: 
        std = 1.0

    normalized_volume = (volume - mean) / std
    
    p_low = np.percentile(normalized_volume[mask], 0.5)
    p_high = np.percentile(normalized_volume[mask], 99.5)
    normalized_volume = np.clip(normalized_volume, p_low, p_high)
    
    # Scale to [0, 1]
    min_val = np.min(normalized_volume)
    max_val = np.max(normalized_volume)
    if max_val - min_val > 1e-5:
        normalized_volume = (normalized_volume - min_val) / (max_val - min_val)
    else:
        normalized_volume = np.zeros_like(normalized_volume)
        
    return normalized_volume.astype(np.float32)


def create_multichannel_mask(seg_slice_np, num_output_classes):
    """
    Converts a single-channel segmentation slice with BraTS labels (0,1,2,4)
    into a multi-channel binary mask (H, W, num_output_classes).
    Mapping:
    - Channel 0: ET (label 4)
    - Channel 1: NCR/NET (label 1)
    - Channel 2: ED (label 2)
    """
    if num_output_classes != 3:
        raise ValueError("This function is configured for NUM_CLASSES = 3 with specific BraTS label mapping.")

    mask_et = (seg_slice_np == 4).astype(np.float32)  # Enhancing Tumor
    mask_ncr_net = (seg_slice_np == 1).astype(np.float32)  # NCR/NET (Tumor Core component)
    mask_ed = (seg_slice_np == 2).astype(np.float32)  # Edema

    return np.stack([mask_et, mask_ncr_net, mask_ed], axis=-1) # Shape (H, W, 3)


def preprocess_and_save_patient_slices(patient_id, patient_data_path, output_img_dir, output_mask_dir):
    """
    Processes one patient's data: loads 3D NIfTI, normalizes,
    extracts 2D slices, creates multi-channel masks, filters, and saves as .npy.
    """
    try:
        # Find the paths to the NIfTI files for the patient
        flair_path = glob.glob(os.path.join(patient_data_path, f"{patient_id}_flair.nii.gz"))[0]
        t1_path = glob.glob(os.path.join(patient_data_path, f"{patient_id}_t1.nii.gz"))[0]
        t1ce_path = glob.glob(os.path.join(patient_data_path, f"{patient_id}_t1ce.nii.gz"))[0]
        t2_path = glob.glob(os.path.join(patient_data_path, f"{patient_id}_t2.nii.gz"))[0]
        seg_path = glob.glob(os.path.join(patient_data_path, f"{patient_id}_seg.nii.gz"))[0]

        # Load NIfTI volumes
        flair_vol = nib.load(flair_path).get_fdata()
        t1_vol = nib.load(t1_path).get_fdata()
        t1ce_vol = nib.load(t1ce_path).get_fdata()
        t2_vol = nib.load(t2_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata().astype(np.uint8) # Ensure segmentation is integer

        # Normalize each modality volume
        flair_norm = normalize_volume(flair_vol)
        t1_norm = normalize_volume(t1_vol)
        t1ce_norm = normalize_volume(t1ce_vol)
        t2_norm = normalize_volume(t2_vol)

        num_saved_slices = 0
        # Iterate through slices (assuming axial, which is usually the 3rd dimension, index 2)
        # BraTS volumes are typically (240, 240, 155)
        for slice_idx in range(flair_norm.shape[2]):
            seg_slice = seg_vol[:, :, slice_idx]

            # Filter: Only keep slices with a significant amount of tumor
            # (sum of all tumor classes: ET, NCR/NET, ED)
            if np.sum(seg_slice > 0) < MIN_TUMOR_PIXELS_THRESHOLD :
                continue

            # Stack modalities for the image slice: (H, W, 4)
            img_slice_2d = np.stack([
                flair_norm[:, :, slice_idx],
                t1_norm[:, :, slice_idx],
                t1ce_norm[:, :, slice_idx],
                t2_norm[:, :, slice_idx]
            ], axis=-1)

            # Create multi-channel binary mask: (H, W, NUM_CLASSES)
            mask_slice_2d_multichannel = create_multichannel_mask(seg_slice, NUM_CLASSES)

            # Save the processed 2D slice and mask as .npy files
            img_filename = os.path.join(output_img_dir, f"{patient_id}_slice{slice_idx:03d}.npy")
            mask_filename = os.path.join(output_mask_dir, f"{patient_id}_slice{slice_idx:03d}.npy")
            
            np.save(img_filename, img_slice_2d)
            np.save(mask_filename, mask_slice_2d_multichannel)
            num_saved_slices += 1
        
        print(f"  Saved {num_saved_slices} slices for patient {patient_id}")
        return num_saved_slices > 0 # Return True if any slice was saved for this patient

    except Exception as e:
        print(f"ERROR processing patient {patient_id}: {e}")
        return False


def main():
    # Create output directories
    train_img_path = os.path.join(OUTPUT_DIR, "train", "images")
    train_mask_path = os.path.join(OUTPUT_DIR, "train", "masks")
    val_img_path = os.path.join(OUTPUT_DIR, "val", "images")
    val_mask_path = os.path.join(OUTPUT_DIR, "val", "masks")

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_mask_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_mask_path, exist_ok=True)

    # Get list of all patient directories from the raw data directory
    # Expected format: BraTS20_Training_001, BraTS20_Training_002, etc.
    patient_dirs = sorted([
        os.path.join(RAW_DATA_DIR, d) for d in os.listdir(RAW_DATA_DIR) 
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and d.startswith("BraTS20_Training_")
    ])

    if not patient_dirs:
        print(f"No patient directories found in {RAW_DATA_DIR} matching the pattern 'BraTS20_Training_XXX'. Please check the RAW_DATA_DIR path.")
        return

    # Shuffle patient directories for random train/val split
    np.random.seed(42) # for reproducibility
    np.random.shuffle(patient_dirs)

    # Split patient list into training and validation sets
    split_idx = int(len(patient_dirs) * TRAIN_VAL_SPLIT_RATIO)
    train_patient_list = patient_dirs[:split_idx]
    val_patient_list = patient_dirs[split_idx:]

    print(f"Total patients: {len(patient_dirs)}")
    print(f"Training patients: {len(train_patient_list)}")
    print(f"Validation patients: {len(val_patient_list)}")

    # Process training data
    print("\nProcessing Training Data...")
    processed_train_patients = 0
    for patient_data_path in train_patient_list:
        patient_id = os.path.basename(patient_data_path)
        if preprocess_and_save_patient_slices(patient_id, patient_data_path, train_img_path, train_mask_path):
            processed_train_patients +=1
    print(f"Finished processing training data. Processed {processed_train_patients} patients successfully.")

    # Process validation data
    print("\nProcessing Validation Data...")
    processed_val_patients = 0
    for patient_data_path in val_patient_list:
        patient_id = os.path.basename(patient_data_path)
        if preprocess_and_save_patient_slices(patient_id, patient_data_path, val_img_path, val_mask_path):
            processed_val_patients += 1
    print(f"Finished processing validation data. Processed {processed_val_patients} patients successfully.")

    print(f"\nPreprocessing complete. Data saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()