import argparse
import json
import os.path as op
from copy import deepcopy

import torch
import cv2
import numpy as np

# os.chdir("..")
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

global _matcher, _PRECISION

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
_PRECISION = 'fp32'  # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can choose model type in ['full', 'opt']
_MODEL_TYPE = 'full'  # 'full' for best quality, 'opt' for best efficiency

def configure_model():
    global _matcher, _PRECISION, _MODEL_TYPE

    # You can also change the default values like thr. and npe (based on input image size)
    if _MODEL_TYPE == 'full':
        config = deepcopy(full_default_cfg)
    elif _MODEL_TYPE == 'opt':
        config = deepcopy(opt_default_cfg)
    else:
        raise Exception(f'Invalid model type: {_MODEL_TYPE}')

    if _PRECISION == 'mp':
        config['mp'] = True
    elif _PRECISION == 'fp16':
        config['half'] = True

    print(config)
    _matcher = LoFTR(config=config)

    _matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
    _matcher = reparameter(_matcher)  # no reparameterization will lead to low performance

    if _PRECISION == 'fp16':
        _matcher = _matcher.half()

    _matcher = _matcher.eval().cuda()


def resize_image_to_multiple_of_32(image: np.ndarray, max_width: int):
    """
    This will limit size and crop (unlike the original example)
    TODO: this assume the image is wide, should handle tall images
    """

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Compute original aspect ratio
    aspect_ratio = original_width / original_height

    max_width_mult = max_width // 32

    # Compute new dimensions as multiples of 32
    width_mult = original_width // 32

    if width_mult > max_width_mult:
        # Max that's a multiple of 32
        width_mult = max_width_mult
    new_width = width_mult * 32

    new_height = int(np.round(new_width / aspect_ratio))

    resized_image = cv2.resize(image, (new_width, new_height))

    # Crop out height
    if resized_image.shape[0] % 32 != 0:
        crop_height = (resized_image.shape[0] // 32) * 32
        print(resized_image.shape[0])
        return resized_image[:crop_height, :]

    return resized_image

def load_images(img0_path: str, img1_path: str) -> tuple[np.ndarray, np.ndarray]:
    global _matcher, _PRECISION

    img0_raw: np.ndarray = cv2.imread(img0_path)
    img1_raw: np.ndarray = cv2.imread(img1_path)

    img0_resized = resize_image_to_multiple_of_32(img0_raw, 1024)
    img1_resized = resize_image_to_multiple_of_32(img1_raw, 1024)

    return img0_resized, img1_resized


def run(img0_raw: np.ndarray, img1_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _PRECISION == 'fp16':
        img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
    else:
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

    batch = {'image0': img0, 'image1': img1}

    # Inference with EfficientLoFTR and get prediction
    with torch.no_grad():
        if _PRECISION == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                _matcher(batch)
        else:
            _matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        return mkpts0, mkpts1, mconf


def draw_matches(img1, img2, points1, points2, conf, thresh, output_path):
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
    h, mask = cv2.findHomography(source_points, target_points)

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


def validate_inputs(img_path: str, img0: str, img1: str):
    # Check if img_path is a valid directory
    if not op.isdir(img_path):
        raise ValueError(f"Error: The path '{img_path}' is not a valid directory.")

    # Check if img0 and img1 are valid files inside img_path
    img0_path = op.join(img_path, img0)
    img1_path = op.join(img_path, img1)

    if not op.isfile(img0_path):
        raise ValueError(f"Error: The file '{img0}' does not exist in '{img_path}'.")
    if not op.isfile(img1_path):
        raise ValueError(f"Error: The file '{img1}' does not exist in '{img_path}'.")

    return img0_path, img1_path


def main():
    """
    _root = '../M3M_WinterBarley_Dataset/90m_RGBMSP_RTK_Overlap7080/Dataset/'
    _img0_path = op.join(_root, 'DJI_20240405154706_0001_D.JPG')
    _img1_path = op.join(_root, 'DJI_20240405154706_0001_MS_G.TIF')
    """
    parser = argparse.ArgumentParser(description="Run EfficientLoFTR on a pair of images and compute homography")
    parser.add_argument("img_path", type=str, help="Path to the directory containing images.")
    parser.add_argument("img0", type=str, help="Name of the first image file.")
    parser.add_argument("img1", type=str, help="Name of the second image file.")
    parser.add_argument("output_path", type=str, help="Path to the output directory")

    args = parser.parse_args()

    img0_path, img1_path = validate_inputs(args.img_path, args.img0, args.img1)
    
    configure_model()
    (img0_resized, img1_resized) = load_images(img0_path, img1_path)
    img0_resized_grey = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2GRAY)
    img1_resized_grey = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    (mkpts0, mkpts1, mconf) = run(img0_resized_grey, img1_resized_grey)

    # Filter by score
    thresh = 0.2
    i = np.where(mconf > thresh)
    mkpts0_filtered = mkpts0[i]  # RGB
    mkpts1_filtered = mkpts1[i]  # Other

    h, corners_dest = compute_homography(source_points=mkpts1_filtered, target_points=mkpts0_filtered, source_shape=img1_resized.shape)
    print(h)
    print(corners_dest)

    resize0_name = op.basename(img0_path)
    resize0_split = op.splitext(resize0_name)
    resize0_name = f'{resize0_split[0]}_resized{resize0_split[1]}'
    resize1_name = op.basename(img1_path)
    resize1_split = op.splitext(resize1_name)
    resize1_name = f'{resize1_split[0]}_resized{resize1_split[1]}'

    cv2.imwrite(resize0_name, img0_resized)
    cv2.imwrite(resize1_name, img1_resized)

    match_name = f'matches_{resize0_split[0]}_{resize1_split[0]}.png'

    draw_matches(img0_resized, img1_resized, mkpts0, mkpts1, mconf, thresh=thresh, output_path=match_name)

    perspective_config = {
        "corners_src": {
            "tl": [0, 0],
            "tr": [img1_resized.shape[1], 0],
            "bl": [0, img1_resized.shape[0]],
            "br": [img1_resized.shape[1], img1_resized.shape[0]],
        },
        "corners_dest": {
            "tl": np.round(corners_dest[0][:2]).astype(int).tolist(),
            "tr": np.round(corners_dest[1][:2]).astype(int).tolist(),
            "bl": np.round(corners_dest[2][:2]).astype(int).tolist(),
            "br": np.round(corners_dest[3][:2]).astype(int).tolist(),
        }
    }

    with open('perspective_config.json', 'w') as f:
        json.dump(perspective_config, f, indent=2)


if __name__ == '__main__':
    main()
