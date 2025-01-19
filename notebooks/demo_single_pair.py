import json
import os
os.chdir("..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

global _matcher, _precision

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
_precision = 'fp32'  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can choose model type in ['full', 'opt']
_model_type = 'full'  # 'full' for best quality, 'opt' for best efficiency

_root = '../M3M_WinterBarley_Dataset/90m_RGBMSP_RTK_Overlap7080/Dataset/'
_img0_pth = os.path.join(_root, 'DJI_20240405154706_0001_D.JPG')
_img1_pth = os.path.join(_root, 'DJI_20240405154706_0001_MS_NIR.TIF')

def configure_model():
    global _matcher, _precision, _model_type

    # You can also change the default values like thr. and npe (based on input image size)
    if _model_type == 'full':
        config = deepcopy(full_default_cfg)
    elif _model_type == 'opt':
        config = deepcopy(opt_default_cfg)
    else:
        raise Exception(f'Invalid model type: {_model_type}')

    if _precision == 'mp':
        config['mp'] = True
    elif _precision == 'fp16':
        config['half'] = True

    print(config)
    _matcher = LoFTR(config=config)

    _matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
    _matcher = reparameter(_matcher)  # no reparameterization will lead to low performance

    if _precision == 'fp16':
        _matcher = _matcher.half()

    _matcher = _matcher.eval().cuda()


def load_images():
    global _matcher, _precision, _img0_pth, _img1_pth

    img0_raw: np.ndarray = cv2.imread(_img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw: np.ndarray = cv2.imread(_img1_pth, cv2.IMREAD_GRAYSCALE)

    # input size should be divisible by 32
    # Scale them down or else we run out of GPU memory
    h0 = int((img0_raw.shape[1] / 4) // 32 * 32)
    w0 = int((img0_raw.shape[0] / 4) // 32 * 32)
    h1 = int((img1_raw.shape[1] / 2) // 32 * 32)
    w1 = int((img1_raw.shape[0] / 2) // 32 * 32)

    img0_raw = cv2.resize(img0_raw, (h0, w0))
    img1_raw = cv2.resize(img1_raw, (h1, w1))

    return img0_raw, img1_raw


def run(img0_raw: np.ndarray, img1_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _precision == 'fp16':
        img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
    else:
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

    batch = {'image0': img0, 'image1': img1}

    # Inference with EfficientLoFTR and get prediction
    with torch.no_grad():
        if _precision == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                _matcher(batch)
        else:
            _matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        return mkpts0, mkpts1, mconf


def draw(img1, img2, points1, points2, conf, thresh, output_path):
    """
    (rewrote original to work without matplotlib)
    Draws two images side by side with lines connecting corresponding points and saves the result.

    Parameters:
        img1 (ndarray): First image as a NumPy array.
        img2 (ndarray): Second image as a NumPy array.
        points1 (ndarray): Array of points in the first image (shape: Nx2).
        points2 (ndarray): Array of points in the second image (shape: Nx2).
        output_path (str): Path to save the resulting image.
    """
    if points1.shape != points2.shape:
        raise ValueError("points1 and points2 must have the same shape.")

    # Convert single-channel images to BGR
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Get dimensions of both images
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    total_width = width1 + width2
    total_height = max(height1, height2)

    # Create a blank canvas to place both images
    canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Place the images on the canvas
    canvas[:height1, :width1] = img1
    canvas[:height2, width1:width1 + width2] = img2

    # Adjust points2 for the shift in the second image
    shifted_points2 = points2.copy()
    shifted_points2[:, 0] += width1  # Shift x-coordinates for points in the second image

    # Draw anti-aliased lines and points
    for i, (p1, p2) in enumerate(zip(points1, shifted_points2)):
        if conf[i] < thresh:
            continue
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        # Draw a red anti-aliased line connecting the points
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        # Draw yellow anti-aliased circles at the points
        cv2.circle(canvas, (x1, y1), 5, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 5, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # Save the resulting image
    cv2.imwrite(output_path, canvas)
    print(f"Image saved to {output_path}")


def compute_homography(source_points: np.ndarray, target_points: np.ndarray, source_shape):
    # Extract source and target points from self.all_pts

    # Compute the homography matrix
    h, _ = cv2.findHomography(source_points, target_points)

    # Get source image dimensions
    height, width = source_shape[:2]

    # Define the source image corners
    corners = np.array([
        [0, 0, 1],  # Top-left
        [width, 0, 1],  # Top-right
        [0, height, 1],  # Bottom-left
        [width, height, 1]  # Bottom-right
    ]).T

    # Transform the corners using the homography matrix
    transformed_corners = h @ corners

    # Normalize to convert from homogeneous coordinates
    normalized_corners = (transformed_corners / transformed_corners[2]).T

    # Return the homography matrix and the transformed corners
    return h, tuple(np.round(normalized_corners).astype(int))

def main():
    configure_model()
    (img0_resized, img1_resized) = load_images()
    (mkpts0, mkpts1, mconf) = run(img0_resized, img1_resized)

    i = np.where(mconf > 0.999)

    mkpts0_filtered = mkpts0[i]  # RGB
    mkpts1_filtered = mkpts1[i]  # Other

    h, corners_dest = compute_homography(source_points=mkpts1_filtered, target_points=mkpts0_filtered, source_shape=img1_resized.shape)
    print(h)
    print(corners_dest)

    draw(img0_resized, img1_resized, mkpts0, mkpts1, mconf, thresh=0.999, output_path='matches.png')

    cv2.imwrite("rgb.jpg", img0_resized)
    cv2.imwrite("nir.jpg", img1_resized)

    perspective_config = {
        "corners_src": {
            "tl": [0, 0],
            "tr": [img0_resized.shape[1], 0],
            "bl": [0, img0_resized.shape[0]],
            "br": [img0_resized.shape[1], img0_resized.shape[0]],
        },
        "corners_dest": {
            "tl": np.round(corners_dest[0][:2]).astype(int).tolist(),
            "tr": np.round(corners_dest[1][:2]).astype(int).tolist(),
            "bl": np.round(corners_dest[2][:2]).astype(int).tolist(),
            "br": np.round(corners_dest[3][:2]).astype(int).tolist(),
        }
    }

    with open('perspective_config.json', 'w') as f:
        json.dump(perspective_config, f)


if __name__ == '__main__':
    main()
