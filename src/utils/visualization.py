import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_eye_extraction(img, left_corners, right_corners, left_eye, right_eye, save_path=None):
    """
    Visualize the eye region extraction process.
    
    Args:
        img: Original input image
        left_corners: Left eye corner coordinates
        right_corners: Right eye corner coordinates
        left_eye: Extracted left eye region
        right_eye: Extracted right eye region
        save_path: Optional path to save the visualization
    """
    # Create a copy of the image for visualization
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw eye corners
    for corner in left_corners:
        cv2.circle(vis_img, tuple(map(int, corner)), 3, (0, 0, 255), -1)
    for corner in right_corners:
        cv2.circle(vis_img, tuple(map(int, corner)), 3, (0, 255, 0), -1)
    
    # Draw lines connecting corners
    cv2.line(vis_img, 
             tuple(map(int, left_corners[0])), 
             tuple(map(int, left_corners[1])), 
             (0, 0, 255), 1)
    cv2.line(vis_img, 
             tuple(map(int, right_corners[0])), 
             tuple(map(int, right_corners[1])), 
             (0, 255, 0), 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    # Plot original image with landmarks
    axes[0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Landmarks')
    axes[0].axis('off')
    
    # Plot left eye region
    axes[1].imshow(left_eye, cmap='gray')
    axes[1].set_title('Left Eye Region')
    axes[1].axis('off')
    
    # Plot right eye region
    axes[2].imshow(right_eye, cmap='gray')
    axes[2].set_title('Right Eye Region')
    axes[2].axis('off')
    
    # Plot both eyes side by side
    combined = np.hstack((left_eye, right_eye))
    axes[3].imshow(combined, cmap='gray')
    axes[3].set_title('Combined Eye Regions')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_batch(batch, save_path=None):
    """
    Visualize a batch of processed eye regions.
    
    Args:
        batch: Dictionary containing 'left_eye' and 'right_eye' tensors
        save_path: Optional path to save the visualization
    """
    # Convert tensors to numpy arrays
    left_eyes = batch['left_eye'].cpu().numpy()
    right_eyes = batch['right_eye'].cpu().numpy()
    
    # Create figure
    batch_size = left_eyes.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(8, 4*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Plot left eye
        axes[i, 0].imshow(left_eyes[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Left Eye {i+1}')
        axes[i, 0].axis('off')
        
        # Plot right eye
        axes[i, 1].imshow(right_eyes[i, 0], cmap='gray')
        axes[i, 1].set_title(f'Right Eye {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_gaze_distribution(gazes, save_path=None):
    """
    Plot the distribution of gaze points.
    
    Args:
        gazes: Numpy array of shape (N, 2) containing gaze coordinates
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(gazes[:, 0], gazes[:, 1], alpha=0.5)
    plt.xlabel('Horizontal Gaze Direction')
    plt.ylabel('Vertical Gaze Direction')
    plt.title('Gaze Point Distribution')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 