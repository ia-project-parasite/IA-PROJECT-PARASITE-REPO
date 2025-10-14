
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

def preprocess_dots_and_rename(root_dir):
    """Invert and binarize all -DOTS.jpg, save as _labels.png, and rename all .jpg to .JPG"""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            # Process -DOTS.jpg
            if fname.endswith('-DOTS.jpg'):
                img = cv2.imread(fpath)
                if img is None:
                    continue
                # Invert
                inv = 255 - img
                # Grayscale
                gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
                # Binarize: all except pure black become white
                _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                # Save as 3-channel
                bw3 = cv2.merge([bw, bw, bw])
                out_name = fname.replace('-DOTS.jpg', '_labels.png')
                out_path = os.path.join(dirpath, out_name)
                cv2.imwrite(out_path, bw3)
            # Rename .jpg to .JPG
            if fname.endswith('.jpg') and not fname.endswith('.JPG'):
                new_name = fname[:-4] + '.JPG'
                new_path = os.path.join(dirpath, new_name)
                os.rename(fpath, new_path)

def find_image_mask_pairs(root_dir):
    """Detect and return all valid pairs of raw image and mask paths"""
    pairs = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        raw_img_path = os.path.join(folder_path, f"{folder}.JPG")
        mask_img_path = os.path.join(folder_path, f"{folder}_labels.png")
        if os.path.exists(raw_img_path) and os.path.exists(mask_img_path):
            print(f"Found pair: {folder}")
            pairs.append((raw_img_path, mask_img_path, folder))
        else:
            print(f"Missing pair: {folder}")
    return pairs

def correct_exposure(img):
    """Exposure correction using CLAHE (adaptive histogram equalization)"""
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    img_lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

def get_bg_mask(adj_img, scale=15):
    """Gaussian blur and subtract to get edge image, generate mask for pure black area"""
    blurred = cv2.GaussianBlur(adj_img, (scale, scale), 0)
    edge = cv2.absdiff(adj_img, blurred)
    # Direct binarization and inversion, keep low value area
    edge_gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY) if len(edge.shape) == 3 else edge
    _, mask = cv2.threshold(edge_gray, 5, 255, cv2.THRESH_BINARY_INV)
    return edge, mask

def get_white_points_and_crop(mask):
    """Get white point coordinates and bounding box, return crop region and points"""
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    points = np.column_stack(np.where(thresh == 255))
    if points.size == 0:
        return None, None
    
    centers = [(int(x), int(y)) for y, x in points]
    return centers, thresh

def split_and_annotate(img_crop, centers, out_dir, out_dir_vis, prefix, n_split=20):
    """Split into n_split squares, annotate which points belong to which tile, save segmentation info"""
    h, w = img_crop.shape[:2]
    side = int(np.sqrt(h * w / n_split))
    n_rows = max(1, h // side)
    n_cols = max(1, w // side)
    imgs_dir = os.path.join(out_dir, 'imgs')
    labels_dir = os.path.join(out_dir, 'labels')
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    for i in range(n_rows):
        for j in range(n_cols):
            x1, y1 = j * side, i * side
            x2, y2 = min((j+1)*side, w), min((i+1)*side, h)
            seg = img_crop[y1:y2, x1:x2]
            # Find points belonging to this seg
            seg_points = []
            for idx, (x, y) in enumerate(centers):
                if x1 <= x < x2 and y1 <= y < y2:
                    seg_points.append({'point_idx': idx, 'x': x, 'y': y})
            seg_name = f"seg_{i}_{j}.jpg"
            seg_path = os.path.join(imgs_dir, seg_name)
            # Save seg without annotation
            cv2.imwrite(seg_path, seg)
            # Save seg with annotation
            if len(seg_points) > 0:
                seg_vis = seg.copy()
                for pt in seg_points:
                    cv2.circle(seg_vis, (pt['x']-x1, pt['y']-y1), 5, (0,0,255), -1)
                seg_vis_path = os.path.join(out_dir_vis, f"seg_{i}_{j}_labels.jpg")
                cv2.imwrite(seg_vis_path, seg_vis)
                # Save csv for this seg
                seg_df = pd.DataFrame(seg_points)
                seg_csv_path = os.path.join(labels_dir, f"seg_{i}_{j}.csv")
                seg_df.to_csv(seg_csv_path, index=False)

def visualize_annotations(img_crop, info, out_path):
    """Generate visualization image for verification"""
    vis = img_crop.copy()
    for item in info:
        cv2.circle(vis, (item['x'], item['y']), 5, (0,0,255), -1)
        cv2.putText(vis, item['seg'], (item['x'], item['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    cv2.imwrite(out_path, vis)

def process_all(root_dir, out_root):
    # 预处理DOTS和重命名
    preprocess_dots_and_rename(root_dir)
    pairs = find_image_mask_pairs(root_dir)
    use_exposure = True  # Exposure correction option, True to use, False to skip
    for raw_img_path, mask_img_path, folder in tqdm(pairs):
        img = cv2.imread(raw_img_path)
        mask = cv2.imread(mask_img_path)
        if use_exposure:
            img_proc = correct_exposure(img)
        else:
            img_proc = img
        centers, thresh = get_white_points_and_crop(mask)
        if centers is None:
            continue
        out_dir = os.path.join(out_root, folder)
        segment_dir = os.path.join(out_dir, 'segment')
        vis_img_dir = os.path.join(out_dir, 'vis')
        
        imgs_dir = os.path.join(segment_dir, 'imgs')
        labels_dir = os.path.join(segment_dir, 'labels')
        os.makedirs(segment_dir, exist_ok=True)
        os.makedirs(vis_img_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    # Save exposure corrected image (if used)
        if use_exposure:
            cv2.imwrite(os.path.join(out_dir, f"{folder}_adj.jpg"), img_proc)
    # Main csv only saves point coordinates and id
        main_csv = pd.DataFrame([{'point_idx': idx, 'x': x, 'y': y} for idx, (x, y) in enumerate(centers)])
        main_csv.to_csv(os.path.join(labels_dir, f"{folder}_points.csv"), index=False)
    # Main image annotation
        vis_img = img_proc.copy()
        for idx, (x, y) in enumerate(centers):
            cv2.circle(vis_img, (x, y), 5, (0,0,255), -1)
            cv2.putText(vis_img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
        cv2.imwrite(os.path.join(vis_img_dir, f"{folder}_vis.jpg"), vis_img)
    # Segmentation csv and images saved in segment subfolder, using exposure corrected image
        split_and_annotate(img_proc, centers, segment_dir, vis_img_dir, folder)

if __name__ == "__main__":
    process_all("labeled_images", "processed_output")