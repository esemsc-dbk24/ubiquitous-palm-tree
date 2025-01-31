import os
import PIL
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from IPython import display
from torch.utils.data import Dataset, DataLoader
from task_3 import make_gif


def task2_plot_event(data_loader, model, output_gif=False, save_gif=False):
    """
    Helper function for visualizing the predictions of a model on a given dataset of events.

    This function visualizes the frames of an event by plotting the input channels (e.g., 3 channels) 
    and the corresponding predicted result for each frame.
    Can output a gif of the frames or display in the notebook

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The DataLoader object that provides batches of the dataset. The dataset should support 
        indexing and provide at least 3 channels for each frame.
    
    model : torch.nn.Module
        The model that will be used to make predictions for each frame. The model should accept
        an input tensor of shape `(batch_size, num_channels, height, width)` and return predictions
        for each frame.
    
    output_gif : bool, optional
        Whether to generate a gif for the frames or not. Default is `False`.
    
    save_gif : bool, optional
        If `True`, the generated gifs will be saved as files. Default is `False`.

    Returns
    -------
    None
        This function does not return any value. It either displays the frames in the notebook 
        or saves the generated gif(s) based on the `output_gif` and `save_gif` flags.
    
    Notes
    -----
    - The function handles frame display by iterating over the dataset in chunks of 36 frames.
    - If `output_gif` is `True`, a temporary `.png` file for each frame is saved, then used to create a gif.
    - Temporary files created for gifs are removed after the gif is generated unless `save_gif` is `True`.
    - The model's output is assumed to be a single channel prediction.

    Dependencies:
    -------------
    this function depends on the make_gif custom function

    """
    def plot_frame(index):
        """
        Helper function for plotting an individual frame from the dataset and its corresponding prediction.
        This function is called by `plot_event` to visualize a single frame's input channels and predicted result side by side.
        
        Parameters
        ----------
        index : int
            The index of the frame to plot from the dataset.

        Returns
        -------
        None
            This function does not return any value.
            It either displays the frame in the notebook or saves the frame as a temporary file if `output_gif` is set to `True`.
        
        Notes
        -----
        - This function retrieves the frame from the dataset, adds a batch dimension, and passes it through the model to get the prediction.
        - The frame’s plot includes the three input channels and the predicted result.

        """

        channels = data_loader.dataset.__getitem__(index)
        channels = channels.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            result = model(channels).squeeze(0).squeeze(0)  # Call the model to predict the result

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Frame: {index}")

        axs[0].imshow(channels[0, 0], cmap="gray"), axs[0].set_title('Channel 1')
        axs[1].imshow(channels[0, 1], cmap="viridis"), axs[1].set_title('Channel 2')
        axs[2].imshow(channels[0, 2], cmap="inferno"), axs[2].set_title('Channel 3')
        axs[3].imshow(result, cmap="turbo"), axs[3].set_title('Predicted Result')

        if output_gif:
            file = f"_temp_{index}.png"
            fig.savefig(file, bbox_inches="tight", dpi=150, pad_inches=0.02, facecolor="white")
            plt.close()
        else:
            plt.show()

    if output_gif:
        num_frames = len(data_loader.dataset)
        num_gifs = num_frames // 36

        for gif_index in range(num_gifs):
            files = []
            for frame_index in range(36):
                index = gif_index * 36 + frame_index
                plot_frame(index)
                files.append(f"_temp_{index}.png")
            im = make_gif(f"output_{gif_index}.gif", files)
            for file in files:
                os.remove(file)
            IPython.display.display(im)
            if not save_gif:
                os.remove(f"output_{gif_index}.gif")
    else:
        for index in range(0, len(data_loader.dataset), 36):
            plot_frame(index)


class Task2DataPreparer:
    """
    A class for preparing event-based image datasets for machine learning models.
    
    This class provides functionality to:
    - Load event IDs and event data from an HDF5 file.
    - Compute normalization parameters (min-max scaling) for images.
    - Resize and normalize image data.
    - Split the dataset into training, validation, and test sets.
    - Create PyTorch data loaders.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing the event data.
    test_size : float, optional
        Proportion of the dataset to be used for testing. Default is 0.2.
    val_size : float, optional
        Proportion of the dataset to be used for validation. Default is 0.1.
    sample_ratio : float, optional
        Ratio of events to be sampled from the dataset. Default is 0.5.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    is_real_data : bool, optional
        Flag to indicate if the data is real data (no VIL channel). Default is False.

    Attributes
    ----------
    norm_params : dict or None
        Dictionary storing min-max normalization parameters for each image type.
        Computed using the training dataset.
    """

    def __init__(self, file_path, test_size=0.2, val_size=0.1, sample_ratio=0.5, random_state=42, is_real_data=False):
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.is_real_data = is_real_data
        self.norm_params = None  # Will be computed later

    def task2_load_event_ids(self):
        """
        Load all event IDs from the HDF5 file.

        Returns
        -------
        list of str
            A list of event IDs available in the dataset.
        """
        with h5py.File(self.file_path, 'r') as f:
            event_ids = list(f.keys())  # Extract event IDs from the file
        return event_ids
    

    def task2_load_event(self, event_id):
        """
        Load the image data of a specific event.

        Parameters
        ----------
        event_id : str
            The ID of the event to be loaded.

        Returns
        -------
        dict
            Dictionary containing image arrays for different channels:
            {'vis': array, 'ir069': array, 'ir107': array, 'vil': array}.
        """
        with h5py.File(self.file_path, 'r') as f:
            if self.is_real_data:
                event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107']}
            else:
                event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil']}
        return event

    def compute_normalization_params(self, event_ids):
        """
        Compute min-max normalization parameters for each image type.

        Parameters
        ----------
        event_ids : list of str
            List of event IDs used to compute normalization parameters.
        
        Returns
        -------
        None
            This function does not return any value. It computes the normalization parameters and stores them in the `norm_params` attribute.
        """
        # Initialize min and max values for each image type
        vis_min, vis_max = float('inf'), float('-inf')
        ir069_min, ir069_max = float('inf'), float('-inf')
        ir107_min, ir107_max = float('inf'), float('-inf')
        vil_min, vil_max = float('inf'), float('-inf')

        # Iterate over events to compute min and max values
        for event_id in event_ids:
            event = self.load_event(event_id)
            vis, ir069, ir107 = event['vis'], event['ir069'], event['ir107']

            vis_min, vis_max = min(vis_min, vis.min()), max(vis_max, vis.max())
            ir069_min, ir069_max = min(ir069_min, ir069.min()), max(ir069_max, ir069.max())
            ir107_min, ir107_max = min(ir107_min, ir107.min()), max(ir107_max, ir107.max())

            if not self.is_real_data:
                vil = event['vil']
                vil_min, vil_max = min(vil_min, vil.min()), max(vil_max, vil.max())

        # Store normalization parameters
        self.norm_params = {
            'vis': (vis_min, vis_max),
            'ir069': (ir069_min, ir069_max),
            'ir107': (ir107_min, ir107_max)
        }

        if not self.is_real_data:
            self.norm_params['vil'] = (vil_min, vil_max)

    def adjust_image_size(self, event):
        """
        Resize and normalize the input images for an event, including VIS, IR069, IR107, and optionally VIL images.
        This function resizes the input images to a target shape of (192, 192) and normalizes them using precomputed 
        min-max values for each image type. If the event contains a VIL image, it is also resized and normalized.

        Parameters
        ----------
        event : dict
            A dictionary containing the original image data for the event. The dictionary must include the keys:
                - 'vis': The VIS image data (numpy array).
                - 'ir069': The IR069 image data (numpy array).
                - 'ir107': The IR107 image data (numpy array).
                - 'vil' (optional): The VIL image data (numpy array). Only required for real data processing.

        Returns
        -------
        tuple of numpy.ndarray
            Resized and normalized images. The tuple includes the following:
                - Resized and normalized VIS image (numpy array).
                - Resized and normalized IR069 image (numpy array).
                - Resized and normalized IR107 image (numpy array).
                - (Optional) Resized and normalized VIL image (numpy array), if the event contains VIL data.

        Notes
        -----
        - If `self.is_real_data` is `True`, only the VIS, IR069, and IR107 images are returned.
        - If `self.is_real_data` is `False`, the VIL image is also resized and normalized, and all four images are returned.
        """
        target_shape = (192, 192)  # Target image size

        # Extract image channels
        X_vis, X_ir069, X_ir107 = event['vis'], event['ir069'], event['ir107']

        # Resize VIS images if needed
        if X_vis.shape[:2] != target_shape:
            X_vis_resized = np.stack([resize(X_vis[:, :, t], target_shape, mode='reflect',
                                             preserve_range=True, anti_aliasing=True) for t in range(X_vis.shape[2])], axis=-1)
        else:
            X_vis_resized = X_vis

        # Normalize images using precomputed min-max values
        X_vis_resized = (X_vis_resized - self.norm_params['vis'][0]) / (self.norm_params['vis'][1] - self.norm_params['vis'][0])
        X_ir069 = (X_ir069 - self.norm_params['ir069'][0]) / (self.norm_params['ir069'][1] - self.norm_params['ir069'][0])
        X_ir107 = (X_ir107 - self.norm_params['ir107'][0]) / (self.norm_params['ir107'][1] - self.norm_params['ir107'][0])

        if self.is_real_data:
            return X_vis_resized, X_ir069, X_ir107
        else:
            y_vil = event['vil']
            # Resize VIL images if needed
            if y_vil.shape[:2] != target_shape:
                y_vil_resized = np.stack([resize(y_vil[:, :, t], target_shape, mode='reflect',
                                                  preserve_range=True, anti_aliasing=True) for t in range(y_vil.shape[2])], axis=-1)
            else:
                y_vil_resized = y_vil

            # Normalize VIL
            y_vil_resized = (y_vil_resized - self.norm_params['vil'][0]) / (self.norm_params['vil'][1] - self.norm_params['vil'][0])

            return X_vis_resized, X_ir069, X_ir107, y_vil_resized

    def prepare_datasets(self):
        """
        Prepare training, validation, and test datasets.

        This function loads event IDs, randomly samples a subset based on `self.sample_ratio`,
        splits the dataset into training, validation, and test sets, and normalizes the data
        using statistics from the training set.

        Returns
        -------
        tuple
            If `self.is_real_data` is True:
                - X_train : np.ndarray
                    Feature data for training set.
                - X_val : np.ndarray
                    Feature data for validation set.
                - X_test : np.ndarray
                    Feature data for test set.
                - norm_params : dict
                    Normalization parameters computed from the training set.
            
            If `self.is_real_data` is False:
                - X_train : np.ndarray
                    Feature data for training set.
                - y_train : np.ndarray
                    Target labels for training set.
                - X_val : np.ndarray
                    Feature data for validation set.
                - y_val : np.ndarray
                    Target labels for validation set.
                - X_test : np.ndarray
                    Feature data for test set.
                - y_test : np.ndarray
                    Target labels for test set.
                - norm_params : dict
                    Normalization parameters computed from the training set.
        """
        
        event_ids = self.load_event_ids()

        # Randomly sample event IDs
        selected_ids = np.random.choice(event_ids, size=int(len(event_ids) * self.sample_ratio), replace=False)

        # Split dataset into train, validation, and test sets
        train_ids, test_ids = train_test_split(selected_ids, test_size=self.test_size, random_state=self.random_state)
        train_ids, val_ids = train_test_split(train_ids, test_size=self.val_size / (1 - self.test_size),
                                              random_state=self.random_state)

        # Compute normalization parameters using training data
        self.compute_normalization_params(train_ids)

        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

        # Load and preprocess data
        for dataset, event_list in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)], 
                                        [train_ids, val_ids, test_ids]):
            for event_id in event_list:
                event = self.load_event(event_id)
                if self.is_real_data:
                    X_vis, X_ir069, X_ir107 = self.adjust_image_size(event)
                    for t in range(X_vis.shape[2]):
                        dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
                else:
                    X_vis, X_ir069, X_ir107, y_vil = self.adjust_image_size(event)
                    for t in range(X_vis.shape[2]):
                        dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
                        dataset[1].append(y_vil[:, :, t])

        if self.is_real_data:
            return np.array(X_train), np.array(X_val), np.array(X_test), self.norm_params
        else:
            return (np.array(X_train), np.array(y_train),
                    np.array(X_val), np.array(y_val),
                    np.array(X_test), np.array(y_test),
                    self.norm_params)
    
    def prepare_surprise_datasets(self):
        """
        Prepare the surprise dataset.

        This function loads all available event IDs, computes normalization parameters
        using the entire dataset, processes the event images, and stores them as a numpy array.

        Returns
        -------
        tuple
            - X_data : np.ndarray
                Feature data for all events in the surprise dataset.
            - norm_params : dict
                Normalization parameters computed from the entire dataset.
        """
        event_ids = self.load_event_ids()

        # Compute normalization parameters using the full dataset
        self.compute_normalization_params(event_ids)

        X_data = []

        # Load and preprocess data
        for event_id in event_ids:
            print(event_id)
            event = self.load_event(event_id)
            X_vis, X_ir069, X_ir107 = self.adjust_image_size(event)
            for t in range(X_vis.shape[2]):
                X_data.append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))

        return np.array(X_data), self.norm_params


    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128, num_workers=8):
        """
        Create PyTorch data loaders for training, validation, and test datasets.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray or None
            Training labels.
        X_val : numpy.ndarray
            Validation data.
        y_val : numpy.ndarray or None
            Validation labels.
        X_test : numpy.ndarray
            Test data.
        y_test : numpy.ndarray or None
            Test labels.
        batch_size : int, optional
            Batch size for data loaders. Default is 128.
        num_workers : int, optional
            Number of workers for data loaders. Default is 8.

        Returns
        -------
        tuple
            Data loaders for training, validation, and test datasets.
        """
        train_dataset = self.WeatherDataset(X_train, y_train)
        val_dataset = self.WeatherDataset(X_val, y_val)
        test_dataset = self.WeatherDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader

    class WeatherDataset(Dataset):
        """
        PyTorch Dataset for handling weather-related image data.

        This dataset is designed to store and process image-based weather data, with optional 
        target values (labels) for supervised learning tasks. The input images are assumed 
        to have a shape of (H, W, C), where:
        - H = Height
        - W = Width
        - C = Number of channels (e.g., different satellite bands)

        The dataset applies the transformations needed to make the data compatible with PyTorch models including:
        - Converting NumPy arrays to PyTorch tensors.        
        - Changing the channel order from (H, W, C) to (C, H, W).
        - Ensuring that all data is stored as `float32` for compatibility with deep learning models.

        Parameters
        ----------
        X : array-like (NumPy array or list of NumPy arrays)
            The input feature data (images). Each image should be in the shape (H, W, C).
        y : array-like (NumPy array or list), optional (default=None)
            The target values corresponding to the images, if available. If `y` is not provided, 
            the dataset operates in an unsupervised mode.

        Attributes
        ----------
        X : np.ndarray
            Processed input feature data as a NumPy array of type `float32`.
        y : np.ndarray or None
            Processed target values as a NumPy array of type `float32` (if provided), otherwise `None`.

        Methods
        -------
        __len__():
            Returns the total number of samples in the dataset.

        __getitem__(idx):
            Retrieves the input image and corresponding label (if available) at the given index.
        """
        def __init__(self, X, y=None):
            self.X = np.array(X, dtype=np.float32)  # Ensure X is a numpy array of type float32
            self.y = np.array(y, dtype=np.float32) if y is not None else None  # Ensure y is a numpy array of type float32

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            """
            Retrieves a sample from the dataset at the specified index.

            Parameters
            ----------
            idx : int
                Index of the sample to retrieve.

            Returns
            -------
            torch.Tensor
                Transformed input image tensor with shape (C, H, W).
            torch.Tensor, optional
                Corresponding target value tensor with shape (1,) if `y` is available.
            """
            # Change the order of dimension (H, W, C) -> (C, H, W)
            x = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)
            if self.y is not None:
                y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)  # added channel dimension
                return x, y
            else:
                return x
            

'---- Old, potentially useful code snippets for future iterations or different models ----'
# class Visualizer:
#     def __init__(self, model, dataloader, norm_params, device='cuda', output_dir='./vis_results'):
#         """
#         可视化工具类初始化
#         Args:
#             model: 训练好的模型
#             dataloader: 数据加载器 (验证集或测试集)
#             norm_params: 归一化参数字典
#             device: 计算设备
#             output_dir: 结果保存目录
#         """
#         self.model = model.to(device)
#         self.dataloader = dataloader
#         self.norm_params = norm_params
#         self.device = device
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)

#     def denormalize(self, image, channel):
#         """反归一化图像数据"""
#         min_val, max_val = self.norm_params[channel]
#         return image * (max_val - min_val) + min_val

#     def _create_frame(self, inputs, target, prediction, frame_idx, event_id):
#         """创建单个帧的可视化（修正维度问题）"""
#         fig, axs = plt.subplots(1, 5, figsize=(20, 10))
#         fig.suptitle(f"Event: {event_id} | Frame: {frame_idx}", fontsize=14)

#         # 确保输入数据维度正确 (H, W, C)
#         input_images = {
#             'vis': inputs[0].squeeze(),   # 去除可能的单通道维度
#             'ir069': inputs[1].squeeze(),
#             'ir107': inputs[2].squeeze()
#         }

#         # 可视化输入通道
#         axs[0].imshow(input_images['vis'], cmap='gray')
#         axs[0].set_title("Input (VIS)")
#         axs[0].axis('off')

#         axs[1].imshow(input_images['ir069'], cmap='gray')
#         axs[1].set_title("Input (IR069)")
#         axs[1].axis('off')

#         axs[2].imshow(input_images['ir107'], cmap='gray')
#         axs[2].set_title("Input (IR107)")
#         axs[2].axis('off')

#         # 处理真实值和预测值（确保是2D数组）
#         vil_true = target.squeeze()  # 去除单通道维度
#         vil_pred = prediction.squeeze()

#         # 真实值
#         axs[3].imshow(vil_true, cmap='viridis')
#         axs[3].set_title("True VIL")
#         axs[3].axis('off')

#         # 预测值
#         axs[4].imshow(vil_pred, cmap='viridis')
#         axs[4].set_title("Predicted VIL")
#         axs[4].axis('off')

#         plt.tight_layout()
#         return fig

#     def generate_prediction_gif(self, event_id, num_frames=36, fps=10):
#         """生成预测结果GIF动画（修正维度处理）"""
#         temp_files = []
#         self.model.eval()

#         # 查找指定事件的数据
#         for batch_idx, (inputs, targets) in enumerate(self.dataloader):
#             inputs = inputs.to(self.device)
#             targets = targets.squeeze(1).cpu().numpy()  # 去除通道维度 (N, 1, H, W) -> (N, H, W)

#             with torch.no_grad():
#                 predictions = self.model(inputs).squeeze(1).cpu().numpy()  # (N, 1, H, W) -> (N, H, W)

#             # 生成各帧图像
#             for frame_idx in range(min(num_frames, inputs.shape[0])):
#                 # 获取输入数据并转换维度 (C, H, W) -> (H, W, C)
#                 input_frame = inputs[frame_idx].cpu().numpy().transpose(1, 2, 0)

#                 fig = self._create_frame(
#                     input_frame.transpose(2, 0, 1),  # 分解为各通道 (C, H, W)
#                     targets[frame_idx],
#                     predictions[frame_idx],
#                     frame_idx,
#                     event_id
#                 )

#                 # 保存临时文件
#                 filename = os.path.join(self.output_dir, f"temp_{event_id}_{frame_idx:03d}.png")
#                 fig.savefig(filename, dpi=100, bbox_inches='tight')
#                 plt.close(fig)
#                 temp_files.append(filename)

#             break  # 假设第一个batch包含目标事件

#         # 生成GIF
#         gif_path = os.path.join(self.output_dir, f"prediction_{event_id}.gif")
#         self._make_gif(gif_path, temp_files, fps)

#         # 清理临时文件
#         for f in temp_files:
#             os.remove(f)

#         return display.Image(filename=gif_path)

#     def plot_loss_curve(self, train_losses, val_losses, save_path=None):
#         """绘制损失曲线并保存"""
#         plt.figure(figsize=(10, 6))
#         plt.plot(train_losses, label='Training Loss')
#         plt.plot(val_losses, label='Validation Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Process Loss Curve')
#         plt.legend()
#         plt.grid(True)

#         if save_path:
#             plt.savefig(os.path.join(self.output_dir, save_path), dpi=150)
#         plt.show()

#     def _make_gif(self, outfile, files, fps=10, loop=0):
#         """生成GIF的辅助函数"""
#         imgs = [PIL.Image.open(file) for file in files]
#         imgs[0].save(
#             fp=outfile,
#             format='GIF',
#             append_images=imgs[1:],
#             save_all=True,
#             duration=int(1000/fps),
#             loop=loop
#         )
#         return outfile

# class DataPreparer:
#     """
#     A class for preparing event-based image datasets for machine learning models.
    
#     This class provides functionality to:
#     - Load event IDs and event data from an HDF5 file.
#     - Compute normalization parameters (min-max scaling) for images.
#     - Resize and normalize image data.
#     - Split the dataset into training, validation, and test sets.

#     Parameters
#     ----------
#     file_path : str
#         Path to the HDF5 file containing the event data.
#     test_size : float, optional
#         Proportion of the dataset to be used for testing. Default is 0.2.
#     val_size : float, optional
#         Proportion of the dataset to be used for validation. Default is 0.1.
#     sample_ratio : float, optional
#         Ratio of events to be sampled from the dataset. Default is 0.5.
#     random_state : int, optional
#         Random seed for reproducibility. Default is 42.
#     is_real_data : bool, optional
#         Flag to indicate if the data is real data (no VIL channel). Default is False.

#     Attributes
#     ----------
#     norm_params : dict or None
#         Dictionary storing min-max normalization parameters for each image type.
#         Computed using the training dataset.
#     """

#     def __init__(self, file_path, test_size=0.2, val_size=0.1, sample_ratio=0.5, random_state=42, is_real_data=False):
#         self.file_path = file_path
#         self.test_size = test_size
#         self.val_size = val_size
#         self.sample_ratio = sample_ratio
#         self.random_state = random_state
#         self.is_real_data = is_real_data
#         self.norm_params = None  # Will be computed later

#     def load_event_ids(self):
#         """
#         Load all event IDs from the HDF5 file.

#         Returns
#         -------
#         list of str
#             A list of event IDs available in the dataset.
#         """
#         with h5py.File(self.file_path, 'r') as f:
#             event_ids = list(f.keys())  # Extract event IDs from the file
#         return event_ids

#     def load_event(self, event_id):
#         """
#         Load the image data of a specific event.

#         Parameters
#         ----------
#         event_id : str
#             The ID of the event to be loaded.

#         Returns
#         -------
#         dict
#             Dictionary containing image arrays for different channels:
#             {'vis': array, 'ir069': array, 'ir107': array, 'vil': array}.
#         """
#         with h5py.File(self.file_path, 'r') as f:
#             if self.is_real_data:
#                 event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107']}
#             else:
#                 event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil']}
#         return event

#     def compute_normalization_params(self, event_ids):
#         """
#         Compute min-max normalization parameters for each image type.

#         Parameters
#         ----------
#         event_ids : list of str
#             List of event IDs used to compute normalization parameters.
#         """
#         # Initialize min and max values for each image type
#         vis_min, vis_max = float('inf'), float('-inf')
#         ir069_min, ir069_max = float('inf'), float('-inf')
#         ir107_min, ir107_max = float('inf'), float('-inf')
#         vil_min, vil_max = float('inf'), float('-inf')

#         # Iterate over events to compute min and max values
#         for event_id in event_ids:
#             event = self.load_event(event_id)
#             vis, ir069, ir107 = event['vis'], event['ir069'], event['ir107']

#             vis_min, vis_max = min(vis_min, vis.min()), max(vis_max, vis.max())
#             ir069_min, ir069_max = min(ir069_min, ir069.min()), max(ir069_max, ir069.max())
#             ir107_min, ir107_max = min(ir107_min, ir107.min()), max(ir107_max, ir107.max())

#             if not self.is_real_data:
#                 vil = event['vil']
#                 vil_min, vil_max = min(vil_min, vil.min()), max(vil_max, vil.max())

#         # Store normalization parameters
#         self.norm_params = {
#             'vis': (vis_min, vis_max),
#             'ir069': (ir069_min, ir069_max),
#             'ir107': (ir107_min, ir107_max)
#         }

#         if not self.is_real_data:
#             self.norm_params['vil'] = (vil_min, vil_max)

#     def adjust_image_size(self, event):
#         """
#         Resize and normalize the input images.

#         Parameters
#         ----------
#         event : dict
#             Dictionary containing the original image data for an event.

#         Returns
#         -------
#         tuple of numpy.ndarray
#             Resized and normalized images for VIS, IR069, IR107, and VIL (if applicable).
#         """
#         target_shape = (192, 192)  # Target image size

#         # Extract image channels
#         X_vis, X_ir069, X_ir107 = event['vis'], event['ir069'], event['ir107']

#         # Resize VIS images if needed
#         if X_vis.shape[:2] != target_shape:
#             X_vis_resized = np.stack([resize(X_vis[:, :, t], target_shape, mode='reflect',
#                                              preserve_range=True, anti_aliasing=True) for t in range(X_vis.shape[2])], axis=-1)
#         else:
#             X_vis_resized = X_vis

#         # Normalize images using precomputed min-max values
#         X_vis_resized = (X_vis_resized - self.norm_params['vis'][0]) / (self.norm_params['vis'][1] - self.norm_params['vis'][0])
#         X_ir069 = (X_ir069 - self.norm_params['ir069'][0]) / (self.norm_params['ir069'][1] - self.norm_params['ir069'][0])
#         X_ir107 = (X_ir107 - self.norm_params['ir107'][0]) / (self.norm_params['ir107'][1] - self.norm_params['ir107'][0])

#         if self.is_real_data:
#             return X_vis_resized, X_ir069, X_ir107
#         else:
#             y_vil = event['vil']
#             # Resize VIL images if needed
#             if y_vil.shape[:2] != target_shape:
#                 y_vil_resized = np.stack([resize(y_vil[:, :, t], target_shape, mode='reflect',
#                                                   preserve_range=True, anti_aliasing=True) for t in range(y_vil.shape[2])], axis=-1)
#             else:
#                 y_vil_resized = y_vil

#             # Normalize VIL
#             y_vil_resized = (y_vil_resized - self.norm_params['vil'][0]) / (self.norm_params['vil'][1] - self.norm_params['vil'][0])

#             return X_vis_resized, X_ir069, X_ir107, y_vil_resized

#     def prepare_datasets(self):
#         """
#         Prepare training, validation, and test datasets.

#         Returns
#         -------
#         tuple
#             Numpy arrays for training, validation, and test sets, along with normalization parameters.
#         """
#         event_ids = self.load_event_ids()

#         # Randomly sample event IDs
#         selected_ids = np.random.choice(event_ids, size=int(len(event_ids) * self.sample_ratio), replace=False)

#         # Split dataset into train, validation, and test sets
#         train_ids, test_ids = train_test_split(selected_ids, test_size=self.test_size, random_state=self.random_state)
#         train_ids, val_ids = train_test_split(train_ids, test_size=self.val_size / (1 - self.test_size),
#                                               random_state=self.random_state)

#         # Compute normalization parameters using training data
#         self.compute_normalization_params(train_ids)

#         X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

#         # Load and preprocess data
#         for dataset, event_list in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)], 
#                                         [train_ids, val_ids, test_ids]):
#             for event_id in event_list:
#                 event = self.load_event(event_id)
#                 if self.is_real_data:
#                     X_vis, X_ir069, X_ir107 = self.adjust_image_size(event)
#                     for t in range(X_vis.shape[2]):
#                         dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
#                 else:
#                     X_vis, X_ir069, X_ir107, y_vil = self.adjust_image_size(event)
#                     for t in range(X_vis.shape[2]):
#                         dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
#                         dataset[1].append(y_vil[:, :, t])

#         if self.is_real_data:
#             return np.array(X_train), np.array(X_val), np.array(X_test), self.norm_params
#         else:
#             return (np.array(X_train), np.array(y_train),
#                     np.array(X_val), np.array(y_val),
#                     np.array(X_test), np.array(y_test),
#                     self.norm_params)