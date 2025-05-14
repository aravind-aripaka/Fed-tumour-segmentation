import h5py
import numpy as np

# Replace with the actual path to one of your .h5 files
# Make sure to use the full correct path based on your `ls` output.
# If your current directory is ~, then the path would be:
# 'BraTS2020_training_data/content/data/volume_XXX_slice_YYY.h5'
# For example:
file_path = 'BraTS2020_training_data/content/data/volume_9_slice_100.h5' # Choose any existing file

try:
    with h5py.File(file_path, 'r') as hf:
        print(f"Inspecting file: {file_path}")
        print("Keys (datasets) in this file:", list(hf.keys()))
        
        # Assuming common keys like 'image' and 'mask' - REPLACE IF DIFFERENT
        if 'image' in hf:
            image_data = hf['image'][:] # The [:] loads the data into a numpy array
            print(f"  Dataset 'image': shape={image_data.shape}, dtype={image_data.dtype}")
        else:
            print("  Dataset 'image' not found. Check available keys.")
            
        if 'mask' in hf:
            mask_data = hf['mask'][:]
            print(f"  Dataset 'mask': shape={mask_data.shape}, dtype={mask_data.dtype}")
            # If mask is integer labels, check unique values:
            if mask_data.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                 print(f"    Unique mask values: {np.unique(mask_data)}")
        else:
            print("  Dataset 'mask' not found. Check available keys.")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")