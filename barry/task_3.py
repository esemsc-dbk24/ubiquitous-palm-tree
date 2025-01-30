# Placeholder
import torch
import h5py
import pandas as pd
import PIL
import IPython
import matplotlib.pyplot as plt
import random
import os
import torch.nn.functional as F


def set_device(device="cuda", idx=0):
    """
    (For use in Google Colab)
    Automatically selects the best available device (GPU/CPU).

    Parameters:
    ----------
    device : str, optional
        Desired device ("cuda" or "cpu"), default is "cuda".
    idx : int, optional
        GPU index to use (default: 0).

    Returns:
    ----------
    str
        The best available device string (e.g., "cuda:0" or "cpu").
    """
    if device != "cpu":
        if torch.cuda.is_available() and torch.cuda.device_count() > idx:
            print(f"CUDA installed! Running on GPU {idx}: {torch.cuda.get_device_name(idx)}")
            device = f"cuda:{idx}"
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"CUDA installed but only {torch.cuda.device_count()} GPU(s) available! Running on GPU 0: {torch.cuda.get_device_name(0)}")
            device = "cuda:0"
        else:
            print("No GPU available! Running on CPU.")
            device = "cpu"
    return device


def load_event(id, project_path) :
    """
    Loads image data for a specific event from an HDF5 file.

    Parameters
    ----------
    id : str
        The identifier of the event to load. This is used to access the specific event within the HDF5 file.
    project_path : str
        The path to the project directory.

    Returns
    -------
    event : dict
        A dictionary mapping image types to their corresponding image data for the given event.
        The keys are:
        - 'vis' : Visible spectrum image data.
        - 'ir069' : Infrared image data at wavelength 069.
        - 'ir107' : Infrared image data at wavelength 107.
        - 'vil' : Visible infrared loop image data.
        - 'lght' : Lightning data.
    """


    with h5py.File(project_path + f'/data/train.h5','r') as f:
        event = {img_type: f[id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
    return event


# Function to calculate basic statistics about the values present in each of the image types
def calculate_image_stats(dict_list, image_types):
    """
    Calculate statistics for the spatial dimensions (192x192 or 384x384) across all channels.

    Parameters
    ----------
        dict_list : list
            List of dictionaries containing tensors.
        image_types : list of str
            Names of tensor dictionaries (e.g., ['ir069', 'ir107', ...]).

    Returns:
    ----------
        pd.DataFrame:
            A DataFrame with spatial statistics for each tensor dictionary.
    """
    # initializing a list to store statistic values in
    stats_values = []

    for dict_name, tensor_dict in zip(image_types, dict_list):
        # Initialize cumulative statistics
        global_min = float('inf')
        global_max = float('-inf')
        total_mean = 0
        total_median = 0
        count = 0
        total_std = 0

        for tensor in tensor_dict.values():
            # Remove the batch dimension (shape becomes [192, 192, 36] or [384, 384, 36])
            tensor = tensor.squeeze(0)

            # Reshaping the 3D tensor into a 2D channel -> appending channel dimension values along the 2nd dimension
            if tensor.shape == (192, 192, 36):
                flattened_tensor = tensor.permute(0, 2, 1).reshape(192, 192 * 36)
            else:
                flattened_tensor = tensor.permute(0, 2, 1).reshape(384, 384 * 36)

            # Update global statistics
            global_min = min(global_min, torch.min(flattened_tensor).item())
            global_max = max(global_max, torch.max(flattened_tensor).item())
            total_mean += torch.mean(flattened_tensor).item()
            total_median += torch.median(flattened_tensor).item()
            total_std += torch.std(flattened_tensor).item()

            count += 1

        # Average the statistics over all tensors
        row = {'image type': dict_name}
        row['XY Value Min'] = global_min
        row['XY Value Max'] = global_max
        row['XY Value Mean'] = total_mean / count if count > 0 else float('nan')
        row['XY Value Median'] = total_median / count if count > 0 else float('nan')
        row['XY Value Std dev'] = total_std / count if count > 0 else float('nan')
        stats_values.append(row)

    return pd.DataFrame(stats_values)

# these functions will plot an event
def make_gif(outfile, files, fps=10, loop=0):
    """
    Helper function for saving GIFs

    Parameters
    ----------
    outfile : str
        The name of the output file.
    files : list
        List of file names. 
    fps : int, optional
        Frames per second (default is 10).
    loop : int, optional
        Number of loops (0 for infinite loop, default is 0).
    
    Returns
    -------
    IPython.display.Image
        An IPython display object representing the GIF.

    """
    imgs = [PIL.Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='gif', append_images=imgs[1:],
                 save_all=True, duration=int(1000/fps), loop=loop)
    im = IPython.display.Image(filename=outfile)
    im.reload()
    return im

def plot_event(id, output_gif=False, save_gif=False):
    """
    Helper function for plotting an event

    Parameters
    ----------
    id : str
        The identifier of the event to plot.
    output_gif : bool, optional
        Whether to output a GIF of the event (default is False).
    save_gif : bool, optional
        Whether to save the GIF file (default is False).

    Returns
    -------
    None
    """
    event = load_event(id)
    t = event["lght"][:,0]# time of lightning strike (in seconds relative to first frame)
    def plot_frame(ti):
        f = (t >= ti*5*60 - 2.5*60) & (t < ti*5*60 + 2.5*60)# find which lightning strikes fall in current frame
        fig,axs = plt.subplots(1,4,figsize=(16,4))
        fig.suptitle(f"Event: {id}, Frame: {ti}, Time: {ti*5} min")
        axs[0].imshow(event["vis"][:,:,ti], vmin=0, vmax=10000, cmap="grey"), axs[0].set_title('Visible')
        axs[1].imshow(event["ir069"][:,:,ti], vmin=-8000, vmax=-1000, cmap="viridis"), axs[1].set_title('Infrared (Water Vapor)')
        axs[2].imshow(event["ir107"][:,:,ti], vmin=-7000, vmax=2000, cmap="inferno"), axs[2].set_title('Infrared (Cloud/Surface Temperature)')
        axs[3].imshow(event["vil"][:,:,ti], vmin=0, vmax=255, cmap="turbo"), axs[3].set_title('Radar (Vertically Integrated Liquid)')
        axs[3].scatter(event["lght"][f,3], event["lght"][f,4], marker="x", s=30, c="tab:red")
        axs[3].set_xlim(0,384), axs[3].set_ylim(384,0)
        if output_gif:
            file = f"_temp_{id}_{ti}.png"
            fig.savefig(file, bbox_inches="tight", dpi=150, pad_inches=0.02, facecolor="white")
            plt.close()
        else:
            plt.show()
    if output_gif:
        for ti in range(36): plot_frame(ti)
        im = make_gif(f"{id}.gif", [f"_temp_{id}_{ti}.png" for ti in range(36)])
        for ti in range(36): os.remove(f"_temp_{id}_{ti}.png")
        IPython.display.display(im)
        if not save_gif: os.remove(f"{id}.gif")
    else:
        plot_frame(0)
        plot_frame(17)
        plot_frame(34)



'---- Old, potentially useful code snippets for future iterations or different models ----'

# def reading_raw_tensors(file_path, image_type, event_ids=None):
#     """
#     Reads and converts raw image data from an HDF5 file into a list of PyTorch tensors.
#     Allows specifying event IDs to load a subset of the data.

#     Parameters
#     ----------
#     file_path : str
#         Path to the HDF5 file containing the image data.
#     image_type : str
#         The type of image data to retrieve from each event.
#         (corresponds to the dataset name within each event in the HDF5 file)
#     event_ids : list, optional
#         List of event IDs to retrieve (default is None, meaning all events are loaded).


#     Returns
#     -------
#     list of torch.Tensor
#         A list of PyTorch tensors, each representing image data from a different event.
#         Each tensor has the shape (m,n, 36), which represents a 36 pictures of 'm' by 'n' pixels capturing 5-minute time segments

#     Notes
#     -----
#     - Assumes the HDF5 file is structured such that each event is a key in the root group.
#     - The function opens the HDF5 file at `file_path` and extracts the specified `image_type` from each event.
#     """
#     tensor_dict = {}

#     with h5py.File(file_path, 'r') as f:

#       all_event_ids = list(f.keys())  # retrieve all the event ids

#       # If event_ids is None, use all available event IDs
#       if event_ids is None:
#           event_ids = all_event_ids

#       # loop through the selected events
#       for event_id in event_ids:
#           dataset_path = f"{event_id}/{image_type}"
#           if dataset_path in f:
#               tensor = torch.tensor(f[dataset_path][:], dtype=torch.float32).unsqueeze(0)

#               # Drop lat/lon if it's the lightning dataset
#               if image_type == 'lght' and tensor.shape[2] == 5:
#                   tensor = tensor[:, :, [0, 3, 4]]  # Keep only [t, x, y]

#               tensor_dict[event_id] = tensor
#           else:
#               # Raise error if the dataset path is missing
#               raise KeyError(f"Dataset path '{dataset_path}' not found in the HDF5 file.")

#     return tensor_dict

# def print_dict_shapes(dataset_dict, dataset_name, max_prints=3):
#     """
#     Prints the shapes of the first few entries in a dataset dictionary.
#     If there is a mismatch, an `AssertionError` is raised.

#     Parameters:
#     ----------
#         dataset_dict : dict
#             A dictionary where each key is an event ID and each value is a tensor.
#         dataset_name : str
#             The name of the dataset (used for printing purposes).
#         max_prints : int, optional
#             The maximum number of entries to print (default: 3).

#     Returns:
#     -------
#         None
#     """
#     print(f"Shape of the {dataset_name} dataset")
#     print_count = 0
#     for event_id, tensor in dataset_dict.items():
#         print(f"Event {event_id} - Shape: {tensor.shape}")
#         print_count += 1
#         if print_count >= max_prints:
#             break

#     first_dataset_name, first_dataset = next(iter(dataset_dict.items()))
#     for event_id, tensor in dataset_dict.items():
#         for other_dataset_name, other_dataset in dataset_dict.items():
#             if other_dataset_name != dataset_name:
#                 assert event_id in other_dataset, (
#                     f"Event {event_id} missing in {other_dataset_name}. "
#                     f"Expected all datasets to have the same event IDs."
#                 )
#                 assert other_dataset[event_id].shape == tensor.shape, (
#                     f"Shape mismatch for event {event_id} in {other_dataset_name}. "
#                     f"Expected shape {tensor.shape}, but got {other_dataset[event_id].shape}."
#                 )
# def pad_lightning_data(lght_dict, N_max, padding_value=-999):
#     """
#     Pads the lightning strike data to ensure all tensors have the same size (N_max,).

#     Handles 1D tensors correctly for `t`, `x`, and `y`.

#     Parameters:
#     ----------
#     lght_dict : dict
#         Dictionary containing event-wise tensors of shape (1, N).
#     N_max : int
#         Maximum number of lightning strikes in any event.
#     padding_value : int, optional
#         Value to use for padding (default: -999).

#     Returns:
#     ----------
#     dict
#         Dictionary with all tensors padded to (1, N_max).
#     """
#     lght_padded = {}

#     for event_id, v in lght_dict.items():
#         v = v.squeeze(0)  # Ensure it's 1D (N,)
#         pad_size = N_max - v.shape[0]  # Compute required padding

#         if pad_size > 0:
#             pad = torch.full((pad_size,), padding_value, dtype=torch.float32)  # Create padding tensor
#             v = torch.cat([v, pad], dim=0)  # Append padding

#         lght_padded[event_id] = v.unsqueeze(0)  # Re-add batch dimension (1, N_max)

#     return lght_padded

# def split_dataset(event_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#     """
#     Split dataset event IDs into train, validation, and test sets.

#     Parameters:
#     ----------
#     event_ids : list
#         List of event IDs to be split.
#     train_ratio : float
#         Proportion of events for training.
#     val_ratio : float
#         Proportion of events for validation.
#     test_ratio : float
#         Proportion of events for testing.

#     Returns:
#     ----------
#     train_ids, val_ids, test_ids : tuple
#         Lists of event IDs for each dataset.
#     """
#     assert train_ratio + val_ratio + test_ratio == 1  #Ratios must sum to 1

#     # Shuffle event IDs
#     random.shuffle(event_ids)

#     # Determine split indices
#     total_events = len(event_ids)
#     train_end = int(total_events * train_ratio)
#     val_end = train_end + int(total_events * val_ratio)

#     # Split into sets
#     train_ids = event_ids[:train_end]
#     val_ids = event_ids[train_end:val_end]
#     test_ids = event_ids[val_end:]

#     return train_ids, val_ids, test_ids


# def create_subset_dicts(event_ids, ir069_dict, ir107_dict, vil_dict, vis_dict, lght_dict):
#     """
#     Create subset dictionaries for a given list of event IDs.

#     Parameters:
#     ----------
#     event_ids : list
#         List of event IDs to extract from the dictionaries.

#     Returns:
#     ----------
#     Dictionary subsets for each dataset.
#     """
#     return (
#         {event: ir069_dict[event] for event in event_ids},
#         {event: ir107_dict[event] for event in event_ids},
#         {event: vil_dict[event] for event in event_ids},
#         {event: vis_dict[event] for event in event_ids},
#         {event: lght_dict[event] for event in event_ids}
#     )

# # Split Lightning Data into `t`, `x`, and `y` Components
# def split_lght_dict(lght_dict):
#     """
#     Splits the lightning dataset into three separate dictionaries for time (`t`), x-coordinates (`x`), and y-coordinates (`y`).

#     Parameters:
#     ----------
#     lght_dict : dict
#         A dictionary where each key is an event ID and each value is a tensor of shape `(1, N, 3)`.
#         The tensor contains lightning strike data, where
#         - [0] is the Time (`t`) in s of the lightning strike.
#         - [1] is the X-coordinate (`x`) for the pixel of a lightning strike.
#         - [2] is the Y-coordinate (`y`) for the pixel of a lightning strike.

#     Returns:
#     ----------
#     tuple of (dict, dict, dict)
#         A tuple containing three dictionaries each mapped to the different dimensions `t`, `x`, and `y` respectively

#     """
#     lght_t_dict, lght_x_dict, lght_y_dict = {}, {}, {}

#     for event_id, tensor in lght_dict.items():
#         lght_t_dict[event_id] = tensor[:, :, 0]  # Extract time column
#         lght_x_dict[event_id] = tensor[:, :, 1]  # Extract x-coordinates
#         lght_y_dict[event_id] = tensor[:, :, 2]  # Extract y-coordinates

#     return lght_t_dict, lght_x_dict, lght_y_dict

# def plot_filtered_histogram(data, title="Filtered Histogram", bins=50, min_val=-300, max_val=300):
#     """
#     Plots a histogram of the given tensor data, filtering values within a specific range.

#     Parameters:
#     - data: torch.Tensor (the data to plot)
#     - title: str (title of the histogram)
#     - bins: int (number of bins for the histogram)
#     - min_val: float (minimum value to include in the histogram)
#     - max_val: float (maximum value to include in the histogram)
#     """
#     if isinstance(data, torch.Tensor):
#         data = data.cpu().numpy()  # Convert to NumPy for Matplotlib

#     # 🔹 Filter data within the range [-300, 300]
#     filtered_data = data[(data >= min_val) & (data <= max_val)]

#     plt.figure(figsize=(8, 5))
#     plt.hist(filtered_data.flatten(), bins=bins, alpha=0.7, edgecolor='black')
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.title(f"{title} ({min_val} to {max_val})")
#     plt.grid(True)
#     plt.show()

# class MinMaxScaler:
#     def __init__(self, use_padding_mask=False, padding_value=None):
#         self.min_vals = {}  # Store min per column
#         self.max_vals = {}  # Store max per column
#         self.use_padding_mask = use_padding_mask
#         self.padding_value = padding_value

#     def fit(self, data):
#         """Compute min/max per column while **ignoring -999** in calculations."""
#         if isinstance(data, dict):  # Handle dictionary case
#             for key, tensor in data.items():
#                 valid_data = tensor[tensor != self.padding_value] if self.use_padding_mask else tensor

#                 if valid_data.numel() > 0:  # Ensure valid data exists
#                     self.min_vals[key] = torch.min(valid_data)
#                     self.max_vals[key] = torch.max(valid_data)
#                 else:
#                     self.min_vals[key] = 0  # Default if no valid data found
#                     self.max_vals[key] = 1

#     def transform(self, data):
#         """Apply min-max scaling while keeping -999 unchanged."""
#         if isinstance(data, dict):
#             transformed_data = {}
#             for key, tensor in data.items():
#                 if key not in self.min_vals or key not in self.max_vals:
#                     raise ValueError(f"Scaler not fitted for key: {key}")

#                 if self.use_padding_mask:
#                     mask = tensor == self.padding_value  # Identify -999 values
#                     tensor = (tensor - self.min_vals[key]) / (self.max_vals[key] - self.min_vals[key])
#                     tensor[mask] = self.padding_value  # Restore -999 values
#                 else:
#                     tensor = (tensor - self.min_vals[key]) / (self.max_vals[key] - self.min_vals[key])

#                 transformed_data[key] = tensor
#             return transformed_data
#         else:
#             raise TypeError("Expected dict of tensors")

#     def inverse_transform(self, data):
#         """Reverse scaling while **preserving -999 as it was originally**."""
#         if isinstance(data, dict):
#             inverse_data = {}
#             for key, tensor in data.items():
#                 if key not in self.min_vals or key not in self.max_vals:
#                     raise ValueError(f"Scaler not fitted for key: {key}")

#                 if self.use_padding_mask:
#                     mask = tensor == self.padding_value
#                     tensor = tensor * (self.max_vals[key] - self.min_vals[key]) + self.min_vals[key]
#                     tensor[mask] = self.padding_value  # Restore -999 after inverse transform
#                 else:
#                     tensor = tensor * (self.max_vals[key] - self.min_vals[key]) + self.min_vals[key]

#                 inverse_data[key] = tensor
#             return inverse_data
#         else:
#             raise TypeError("Expected dict of tensors")


# class StandardScaler:
#     def __init__(self, use_padding_mask=False, padding_value=None):
#         self.means = {}  # Store mean per column
#         self.stds = {}  # Store std per column
#         self.use_padding_mask = use_padding_mask
#         self.padding_value = padding_value

#     def fit(self, data):
#         """Compute and store mean/std per column."""
#         if isinstance(data, dict):  # Handle dictionary case
#             for key, tensor in data.items():
#                 if self.use_padding_mask:
#                     mask = tensor != self.padding_value
#                     valid_data = tensor[mask]
#                 else:
#                     valid_data = tensor

#                 self.means[key] = torch.mean(valid_data)
#                 self.stds[key] = torch.std(valid_data)
#         else:  # Handle single tensor case (e.g., lght_data)
#             if self.use_padding_mask:
#                 mask = data != self.padding_value
#                 valid_data = data[mask]
#             else:
#                 valid_data = data

#             self.means["global"] = torch.mean(valid_data)
#             self.stds["global"] = torch.std(valid_data)

#     def transform(self, data):
#         """Apply standardization per column while ignoring masked values."""
#         if not self.means or not self.stds:
#             raise ValueError("Scaler has not been fitted yet.")

#         if isinstance(data, dict):
#             transformed_data = {}
#             for key, tensor in data.items():
#                 if key not in self.means or key not in self.stds:
#                     raise ValueError(f"Scaler has not been fitted for column: {key}")

#                 if self.use_padding_mask:
#                     mask = tensor == self.padding_value  # Identify masked values
#                     standardized_tensor = (tensor - self.means[key]) / self.stds[key]
#                     tensor[~mask] = standardized_tensor[~mask]  # Only update non-masked values
#                 else:
#                     tensor = (tensor - self.means[key]) / self.stds[key]

#                 transformed_data[key] = tensor
#             return transformed_data
#         else:
#             if "global" not in self.means or "global" not in self.stds:
#                 raise ValueError("Scaler has not been fitted for the global tensor.")

#             if self.use_padding_mask:
#                 mask = data == self.padding_value
#                 standardized_data = (data - self.means["global"]) / self.stds["global"]
#                 data[~mask] = standardized_data[~mask]  # Only modify unmasked values
#             else:
#                 data = (data - self.means["global"]) / self.stds["global"]

#             return data

#     def inverse_transform(self, data):
#         """Reverse the standardization transformation."""
#         if not self.means or not self.stds:
#             raise ValueError("Scaler has not been fitted yet.")

#         if isinstance(data, dict):
#             inverse_data = {}
#             for key, tensor in data.items():
#                 if key not in self.means or key not in self.stds:
#                     raise ValueError(f"Scaler has not been fitted for column: {key}")

#                 if self.use_padding_mask:
#                     mask = tensor != self.padding_value
#                     tensor[mask] = tensor[mask] * self.stds[key] + self.means[key]
#                 else:
#                     tensor = tensor * self.stds[key] + self.means[key]

#                 inverse_data[key] = tensor
#             return inverse_data
#         else:
#             if "global" not in self.means or "global" not in self.stds:
#                 raise ValueError("Scaler has not been fitted for the global tensor.")

#             if self.use_padding_mask:
#                 mask = data != self.padding_value
#                 data[mask] = data[mask] * self.stds["global"] + self.means["global"]
#             else:
#                 data = data * self.stds["global"] + self.means["global"]
#             return data

# class LogTransform:
#     def fit(self, data):
#         """Determine the minimum value for shifting during log transformation."""

#         # 🔹 If data is a dict, extract the tensor (assumes single key)
#         if isinstance(data, dict):
#             data = list(data.values())[0]  # Extract tensor

#         # 🔹 Ensure data is a tensor
#         if not isinstance(data, torch.Tensor):
#             raise TypeError(f"Expected torch.Tensor, but got {type(data)}")

#         self.min_shift = torch.min(data).item()  # Get minimum value
#         if self.min_shift < 0:
#             self.min_shift = abs(self.min_shift) + 1e-6  # Ensure positive shift
#         else:
#             self.min_shift = 0  # No shift needed if min is already ≥ 0

#     def __call__(self, data):
#         """Apply log transformation while ensuring non-negativity."""
#         if not hasattr(self, "min_shift"):
#             raise ValueError("LogTransform must be fitted before applying the transformation.")

#         shifted_data = data + self.min_shift  # Shift to make non-negative
#         return torch.log1p(shifted_data)  # log(1 + x)

#     def inverse_transform(self, data):
#         """Reverse the log transformation."""
#         if not hasattr(self, "min_shift"):
#             raise ValueError("LogTransform must be fitted before applying the inverse transformation.")

#         return torch.expm1(data) - self.min_shift  # Reverse shift after expm1


# class ResizeDown3D:
#     def __init__(self, target_size=(192, 192)):
#         self.target_size = target_size

#     def __call__(self, data):
#         """
#         Resizes only the spatial dimensions (H, W) while keeping T unchanged as they are constant for the non-target img_types
#         Ensures batch dimension is retained.
#         """
#         # 🔍 Check if batch dimension is present
#         batch_dim = False
#         if data.dim() == 4:  # Shape: (1, H, W, T)
#             batch_dim = True
#         elif data.dim() == 3:  # Shape: (H, W, T)
#             data = data.unsqueeze(0)  # Add batch dim
#             batch_dim = False
#         else:
#             raise ValueError(f"Unexpected shape {data.shape}, expected (H, W, T) or (1, H, W, T).")

#         # Convert (1, H, W, T) -> (1, T, H, W) for interpolation
#         data = data.permute(0, 3, 1, 2)  # (1, T, H, W)

#         # Apply resizing
#         resized_data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)

#         # Convert back to (1, H, W, T)
#         resized_data = resized_data.permute(0, 2, 3, 1)  # (1, H, W, T)

#         # Remove batch dim if it wasn't there originally
#         return resized_data if batch_dim else resized_data.squeeze(0)


# class ResizeUp3D:
#     def __init__(self, target_size=(384, 384)):
#         self.target_size = target_size

#     def __call__(self, data):
#         """
#         Resizes only the spatial dimensions (H, W) while keeping T unchanged.
#         Ensures batch dimension is retained.

#         Parameters:
#         ----------
#         data : torch.Tensor
#             Input data to resize.

#         Returns:   
#         ----------
#         torch.Tensor
#             Resized data tensor.    

#         """
#         batch_dim = False
#         if data.dim() == 4:  # Shape: (1, H, W, T)
#             batch_dim = True
#         elif data.dim() == 3:  # Shape: (H, W, T)
#             data = data.unsqueeze(0)  # Add batch dim
#             batch_dim = False
#         else:
#             raise ValueError(f"Unexpected shape {data.shape}, expected (H, W, T) or (1, H, W, T).")

#         # Convert (1, H, W, T) -> (1, T, H, W) for interpolation
#         data = data.permute(0, 3, 1, 2)  # (1, T, H, W)

#         # Apply resizing
#         resized_data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)

#         # Convert back to (1, H, W, T)
#         resized_data = resized_data.permute(0, 2, 3, 1)  # (1, H, W, T)

#         # Remove batch dim if it wasn't there originally
#         return resized_data if batch_dim else resized_data.squeeze(0)

# class Pipeline:
#     def __init__(self, steps):
#         """
#         Create a sequential pipeline of transformations.

#         Parameters:
#         ----------
#         steps : list of tuples
#             Each tuple is of the form (name, transform), where `name` is a string
#             describing the step, and `transform` is a callable object or an object
#             with a `transform` method.
#         """
#         self.steps = steps

#     def __call__(self, data):
#         """
#         Apply all transformations sequentially.

#         Parameters:
#         ----------
#         data : torch.Tensor
#             Input data to transform.

#         Returns:
#         ----------
#         Transformed data.
#         """
#         for name, transform in self.steps:
#             if hasattr(transform, 'transform'):
#                 # If the transform has a `transform` method, use it
#                 data = transform.transform(data)
#             else:
#                 # Otherwise, assume the transform is callable
#                 data = transform(data)
#         return data

#     def inverse_transform(self, data):
#         """
#         Undo all transformations in reverse order.

#         Parameters:
#         ----------
#         data : torch.Tensor
#             Transformed data to inverse.

#         Returns:
#         ----------
#         Original scale data.
#         """
#         for name, transform in reversed(self.steps):
#             if hasattr(transform, 'inverse_transform'):
#                 data = transform.inverse_transform(data)
#         return data


# class ColumnTransformer:
#     def __init__(self, transformers):
#         self.transformers = transformers
#         self.fitted = False  # Track if scalers are fitted

#     def fit(self, img_data, lght_data):
#         """
#         Fit scalers and transformations using training data.
        
#         Parameters:
#         ----------
#         img_data : dict
#             Dictionary containing image data tensors.
#         lght_data : dict
#             Dictionary containing lightning data tensors.
        
#         Returns:
#         ----------
#         None
        
#         """
#         print("Fitting transformers...")  # Debugging
#         self.fitted_transformers = {}  # Store fitted scalers

#         for name, transform, columns in self.transformers:
#             for column in columns:
#                 if column in img_data and hasattr(transform, 'fit'):
#                     print(f"Fitting {name} for column: {column}")  # Debugging
#                     transform.fit({'global': img_data[column]})  # Store as global
#                     self.fitted_transformers[column] = transform  # Store fitted scaler

#                 elif column.startswith('lght') and hasattr(transform, 'fit'):
#                     if column in lght_data:
#                         print(f"Fitting {name} for lightning data: {column}")  # Debugging
#                         transform.fit({'global': lght_data[column]})  # Store as global
#                         self.fitted_transformers[column] = transform  # Store fitted scaler
#                     else:
#                         print(f"Warning: Missing {column} in lght_data during fitting.")

#         self.fitted = True
#         print("All transformers fitted.\n")  # Debugging

#     def __call__(self, img_data, lght_data):
#         """
#         Apply transformations.   
        
#         Parameters:
#         ----------
#         img_data : dict
#             Dictionary containing image data tensors.

#         lght_data : dict
#             Dictionary containing lightning data tensors.
        
#         Returns:
#         ----------
#         dict : dict, dict
#             Transformed image and lightning data.
        
#         """


#         if not self.fitted:
#             raise ValueError("ColumnTransformer must be fitted before use.")   # Debugging

#         if not isinstance(lght_data, dict):
#             raise TypeError(f"Expected `lght_data` to be a dict, but got {type(lght_data)}") # Debugging

#         for name, transform, columns in self.transformers:
#             for column in columns:
#                 if column in img_data:
#                     if hasattr(transform, 'transform'):
#                         img_data[column] = transform.transform({column: img_data[column]})[column]
#                     else:
#                         img_data[column] = transform(img_data[column])

#                 elif column.startswith('lght'):
#                     if column not in lght_data:
#                         raise KeyError(f"Missing expected key {column} in lght_data")

#                     if hasattr(transform, 'transform'):
#                         lght_data[column] = transform.transform({column: lght_data[column]})[column]
#                     else:
#                         lght_data[column] = transform(lght_data[column])

#         return img_data, lght_data

#     def inverse_transform(self, img_data, lght_data):
#         """
#         Reverse transformations for visualization or interpretation.

#         Parameters:
#         ----------
#         img_data : dict
#             Dictionary containing image data tensors.
#         lght_data : dict
#             Dictionary containing lightning data tensors.

#         Returns:
#         ----------
#         dict, dict
#             Inverse-transformed image and lightning data.
        
#         """
#         if not self.fitted:
#             raise ValueError("ColumnTransformer must be fitted before applying inverse_transform.")

#         print("Applying inverse transformations...")  # Debugging

#         for name, transform, columns in self.transformers:
#             for column in columns:
#                 if column in img_data and hasattr(transform, 'inverse_transform'):
#                     img_data[column] = transform.inverse_transform({'global': img_data[column]})['global']

#                 elif column.startswith('lght') and hasattr(transform, 'inverse_transform'):
#                     if column in lght_data:
#                         lght_data[column] = transform.inverse_transform({'global': lght_data[column]})['global']
#                     else:
#                         print(f"Warning: Skipping {column} in inverse_transform (not found in lght_data).")

#         print("Inverse transformations applied successfully!\n")  # Debugging
#         return img_data, lght_data

# def inverse_transform_predictions(event_predictions, transformer):
#     """
#     Inverse transforms lightning strike predictions using the fitted transformer.
#     Handles variable-sized predictions by processing each event separately.

#     Parameters:
#     ----------
#     event_predictions : dict
#         dictionary containing event-wise lightning strike predictions.
#     transformer : ColumnTransformer
#         transformer object used for data transformation.


#     Returns:
#     ----------
#     dict
#         dictionary containing event-wise inverse-transformed lightning strike predictions.
#     """

#     if not isinstance(event_predictions, dict):
#         raise TypeError(f"Expected event_predictions to be a dict, but got {type(event_predictions)}")

#     print("\nConverting event_predictions for inverse transformation...") # Debugging

#     inverse_event_predictions = {}

#     for event_id, tensor in event_predictions.items():
#         print(f"Processing Event {event_id} with shape {tensor.shape}...")

#         # Ensure tensor has the correct shape
#         if tensor.ndim != 2 or tensor.shape[1] != 3:
#             raise ValueError(f"Unexpected shape for event {event_id}: {tensor.shape}, expected (N, 3)")

#         # Prepare event-specific dictionaries
#         lght_data = {
#             "lght_t": {"global": tensor[:, 0].unsqueeze(0)},  # Add batch dim (1, N)
#             "lght_x": {"global": tensor[:, 1].unsqueeze(0)},
#             "lght_y": {"global": tensor[:, 2].unsqueeze(0)},
#         }

#         # Apply inverse transformation separately per event
#         _, transformed_lght_t = transformer.inverse_transform({}, lght_data["lght_t"])
#         _, transformed_lght_x = transformer.inverse_transform({}, lght_data["lght_x"])
#         _, transformed_lght_y = transformer.inverse_transform({}, lght_data["lght_y"])

#         # Stack them back together (WITHOUT changing event size)
#         inverse_event_predictions[event_id] = torch.stack([
#             transformed_lght_t["global"].squeeze(0),
#             transformed_lght_x["global"].squeeze(0),
#             transformed_lght_y["global"].squeeze(0),
#         ], dim=1)

#         print(f"Finished processing Event {event_id}, new shape: {inverse_event_predictions[event_id].shape}")

#     return inverse_event_predictions
