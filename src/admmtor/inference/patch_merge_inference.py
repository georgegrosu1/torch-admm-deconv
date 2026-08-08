import torch
import torch.nn.functional as F
from admmtor.eprocessing.dataload import ImageDataset


def get_2d_hanning_window(window_size):
    """Generates a 2D Hanning window for patch blending."""
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
        
    # Create 1D Hanning windows
    window_y = torch.hann_window(window_size[0], periodic=False)
    window_x = torch.hann_window(window_size[1], periodic=False)
    
    # Outer product to get a 2D window
    window_2d = window_y.unsqueeze(1) * window_x.unsqueeze(0)
    
    # Add batch and channel dimensions (1, 1, H, W) for easy broadcasting
    return window_2d.unsqueeze(0).unsqueeze(0)

def patch_based_inference(image, model, patch_size=128, stride=64):
    """
    Performs overlapping patch-based inference with windowed blending.
    
    Args:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W)
        model (callable): The neural network model
        patch_size (int or tuple): Size of the patches (e.g., 128)
        stride (int or tuple): Stride between patches (e.g., 64)
        
    Returns:
        torch.Tensor: The denoised full-resolution image (B, C, H, W)
    """
    B, C, H, W = image.shape
    device = image.device
    
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride)
        
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    # 1. Pad the image so the sliding window doesn't miss the right/bottom edges
    pad_h = (stride_h - (H - patch_h) % stride_h) % stride_h
    pad_w = (stride_w - (W - patch_w) % stride_w) % stride_w
    
    # Reflection padding is usually best for image edges in denoising tasks
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect').to(device)
    _, _, H_pad, W_pad = padded_image.shape
    
    # 2. Initialize accumulation buffers
    output_buffer = torch.zeros_like(padded_image).to(device)
    weight_buffer = torch.zeros_like(padded_image).to(device)
    
    # 3. Create the 2D weighting window and move to the same device as the image
    window = get_2d_hanning_window((patch_h, patch_w)).to(device)
    
    # 4. Extract, process, and blend patches
    with torch.inference_mode():
        for y in range(0, H_pad - patch_h + 1, stride_h):
            for x in range(0, W_pad - patch_w + 1, stride_w):
                
                # Extract the patch
                patch = padded_image[:, :, y:y+patch_h, x:x+patch_w]
                
                # Run the model
                pred_patch = model(patch)
                pred_patch = pred_patch.detach()
                
                # Multiply by the blending window
                weighted_patch = pred_patch * window
                
                # Accumulate the weighted predictions and the weights
                output_buffer[:, :, y:y+patch_h, x:x+patch_w] += weighted_patch
                weight_buffer[:, :, y:y+patch_h, x:x+patch_w] += window

    # 5. Normalize by the accumulated weights
    # Add a tiny epsilon (1e-8) to prevent division by zero in case of unexpected edge gaps
    output_buffer = output_buffer / (weight_buffer + 1e-8)
    
    # 6. Crop back to the original image resolution
    final_output = output_buffer[:, :, :H, :W]
    
    return final_output.detach().cpu()


class PatchMergeInference:
    def __init__(self):
        pass

    def infer(self, model, image: torch.Tensor, patch_size: int=128, stride: int=64) -> torch.Tensor:
        """
        Perform inference on a single image using patch-based processing.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Denoised image tensor of shape (B, C, H, W)
        """
        return patch_based_inference(image, model, patch_size, stride)