#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, math, csv, argparse
import numpy as np
import cv2

# ========== Basic Utilities ==========
def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x, dtype=np.float32)

def clahe01(img01, clip=2.0, tile=8):
    u8 = (normalize01(img01) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(u8)
    return normalize01(out)

def smooth_edge_preserve(img01, d=5, sigmaColor=20, sigmaSpace=7):
    u8 = (normalize01(img01) * 255).astype(np.uint8)
    sm = cv2.bilateralFilter(u8, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return normalize01(sm)

def draw_points(img_bgr, points, color=(0,0,255), radius=5, thickness=2):
    out = img_bgr.copy()
    for (x,y) in points:
        cv2.circle(out, (int(x),int(y)), radius, color, thickness, lineType=cv2.LINE_AA)
    return out

def load_gt_points(csv_path):
    pts = []
    if not os.path.exists(csv_path): return pts
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # Expected columns: point_idx, x, y
        for r in reader:
            x = float(r['x']); y = float(r['y'])
            pts.append((x,y))
    return pts

def montage_horiz(images, gap=8):
    # Input images can be grayscale or color; auto align height
    h = max(im.shape[0] for im in images)
    ws = []
    ims = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h), interpolation=cv2.INTER_AREA)
        ims.append(im); ws.append(im.shape[1])
    c = sum(ws) + gap*(len(images)-1)
    canvas = np.zeros((h, c, 3), np.uint8)
    x = 0
    for im in ims:
        w = im.shape[1]
        canvas[:, x:x+w] = im
        x += w + gap
    return canvas


def _compute_segment_offset_from_image(image_path, n_split=20):
    """Parse seg_i_j and compute the top-left offset (x1,y1) of the tile in the original cropped image."""
    base = os.path.basename(image_path)
    import re
    m = re.search(r"seg_(\d+)_(\d+)", base)
    if not m:
        return 0, 0
    i = int(m.group(1)); j = int(m.group(2))
    segment_dir = os.path.dirname(os.path.dirname(image_path))
    folder_dir = os.path.dirname(segment_dir)
    folder_name = os.path.basename(folder_dir)
    adj_img_path = os.path.join(folder_dir, f"{folder_name}_adj.jpg")
    if not os.path.exists(adj_img_path):
        alt = os.path.join(folder_dir, f"{folder_name}_adj.JPG")
        if os.path.exists(alt):
            adj_img_path = alt
        else:
            return 0, 0
    img = cv2.imread(adj_img_path)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    side = int(np.sqrt((h * w) / float(n_split))) if n_split > 0 else 0
    x1 = j * side
    y1 = i * side
    return x1, y1


def _convert_gt_points_to_local(gt_pts, image_path, n_split=20):
    tile = cv2.imread(image_path)
    if tile is None:
        return []
    th, tw = tile.shape[:2]
    x1, y1 = _compute_segment_offset_from_image(image_path, n_split=n_split)
    local = []
    for (x, y) in gt_pts:
        x_local = x - x1
        y_local = y - y1
        if 0 <= x_local < tw and 0 <= y_local < th:
            local.append((x_local, y_local))
    return local

# ========== Morphology · Ring Enhancement ==========
def ring_enhance_from_L(L01, kernel_inner=9, kernel_outer=25, blur_ksize=5):
    """
    Apply Top-hat + Black-hat on the L channel to enhance ring-like structures
    with bright center and dark border.
    kernel_inner ~ parasite short-side in pixels (recommend ~10)
    kernel_outer ~ overall scale including dark border (recommend ~25)
    """
    L8 = (normalize01(L01) * 255).astype(np.uint8)
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_inner, kernel_inner))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_outer, kernel_outer))
    tophat   = cv2.morphologyEx(L8, cv2.MORPH_TOPHAT,   k_in)
    blackhat = cv2.morphologyEx(L8, cv2.MORPH_BLACKHAT, k_out)
    ring = cv2.addWeighted(tophat, 1.0, blackhat, 1.2, 0)
    if blur_ksize and blur_ksize > 1:
        ring = cv2.GaussianBlur(ring, (blur_ksize, blur_ksize), 0)
    return normalize01(ring), normalize01(tophat), normalize01(blackhat)

def adaptive_binary(u8, block=51, offset=-3):
    # Adaptive threshold (robust to non-uniform illumination), input must be 8-bit
    block = int(block) if int(block)%2==1 else int(block)+1  # must be odd
    th = cv2.adaptiveThreshold(u8, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               block, offset)
    return th

# ========== Evaluation ==========
def match_points_to_centers(gt_pts, centers, radius=10):
    gt_flags = [False]*len(gt_pts)
    det_flags = [False]*len(centers)
    for j, c in enumerate(centers):
        cx, cy = c
        if not gt_pts: break
        dists = [math.hypot(cx-x, cy-y) for (x,y) in gt_pts]
        i = int(np.argmin(dists))
        if dists[i] <= radius and not gt_flags[i]:
            gt_flags[i] = True
            det_flags[j] = True
    tp = sum(gt_flags)
    fp = len(centers) - tp
    fn = len(gt_pts) - tp
    prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
    return tp, fp, fn, prec, rec

# ========== Core Detection ==========
def detect_parasites_by_L_high_S_low(
    bgr,
    ts=0.5, tl=0.6,                      # 仅在使用旧R/条件时有效
    area_range=(30,1200), ar_range=(1.0,10.0),
    do_watershed=False,
    # 新增参数：
    kernel_inner=9, kernel_outer=25,
    wL=0.7, wS=0.2, wB=0.1,
    use_adaptive=True, th_block=51, th_offset=-3
):
    # 1) BGR -> HLS
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    H, L, S = [normalize01(hls[..., i]) for i in range(3)]

    # 2) Preprocessing: enhance S, light smoothing; normalize L
    S_eq = clahe01(S, clip=2.0, tile=8)
    S_eq = smooth_edge_preserve(S_eq, d=5, sigmaColor=20, sigmaSpace=7)
    Ln   = normalize01(L)

    # 3) Ring enhancement (L primary; 1-S as auxiliary)
    ring_L, th_L, bh_L = ring_enhance_from_L(Ln, kernel_inner=kernel_inner, kernel_outer=kernel_outer, blur_ksize=5)
    ring_S, _, _       = ring_enhance_from_L(1.0 - S_eq, kernel_inner=max(7, kernel_inner-2), kernel_outer=kernel_outer, blur_ksize=5)

    # Legacy 'bright and desaturated' saliency map used as an auxiliary factor
    R_base = normalize01((1.0 - S_eq) * Ln)
    # 组合显著图（以环形为主）
    R_mix  = normalize01( wL*ring_L + wS*ring_S + wB*R_base )

    # 4) Thresholding: prefer adaptive, fallback to Otsu
    u8 = (R_mix * 255).astype(np.uint8)
    if use_adaptive:
        mask = adaptive_binary(u8, block=th_block, offset=th_offset) // 255
    else:
        _, thr = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (u8 > _).astype(np.uint8)

    # Optional extra condition: if many false positives remain, you can
    # further require (S < ts) & (L > tl)
    # cond = ((S_eq < ts) & (Ln > tl)).astype(np.uint8)
    # mask = (mask & cond)

    # 5) Morphological cleaning
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

    seg = mask
    if do_watershed:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        sure_fg = (dist > 0.4*dist.max()).astype(np.uint8)  # slightly relaxed to avoid removing small objects
        unknown = cv2.subtract(mask, sure_fg)
        num_mk, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==1] = 0
        markers = cv2.watershed(cv2.blur(bgr,(3,3)), markers)
        seg = (markers > 1).astype(np.uint8)

    # 6) Geometric filtering + counting
    n, labels, stats, cents = cv2.connectedComponentsWithStats(seg, connectivity=8)
    boxes, centers = [], []
    final_mask = np.zeros_like(seg)
    for i in range(1, n):
        x,y,w,h, A = stats[i]
        ar = max(w,h) / (min(w,h)+1e-6)
        if not (area_range[0] <= A <= area_range[1]): continue
        if not (ar_range[0]   <= ar <= ar_range[1]):   continue
        final_mask[labels==i] = 1
        boxes.append((x,y,w,h))
        centers.append((float(cents[i][0]), float(cents[i][1])))

    return {
        'H':H, 'L':Ln, 'S':S_eq,
        'R':R_base,                 # legacy R (bright and desaturated)
        'ring_L': ring_L,           # ring response from L channel
        'ring_mix': R_mix,          # mixed saliency map used for thresholding
        'mask_raw':mask, 'mask_final':final_mask,
        'boxes':boxes, 'centers':centers
    }

# ========== Pipeline ==========
def run_pipeline(
    images_dir, csv_dir, out_dir,
    ts=0.5, tl=0.6,
    area_range=(30,1200), ar_range=(1.0,10.0),
    match_radius=12, draw_gt=True,
    do_watershed=False,
    kernel_inner=9, kernel_outer=25,
    ring_wL=0.7, ring_wS=0.2, ring_wB=0.1,
    use_adaptive=True, th_block=51, th_offset=-3
):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    for ip in img_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        bgr = cv2.imread(ip)
        if bgr is None: 
            print(f"[WARN] cannot read image: {ip}")
            continue

        csv_path = os.path.join(csv_dir, name + ".csv")
        gt_pts = load_gt_points(csv_path)
        # If this is a segment tile, convert GT coordinates to tile-local coordinates
        if os.path.basename(ip).startswith('seg_') and gt_pts:
            gt_pts = _convert_gt_points_to_local(gt_pts, ip)

        res = detect_parasites_by_L_high_S_low(
            bgr, ts=ts, tl=tl,
            area_range=area_range, ar_range=ar_range,
            do_watershed=do_watershed,
            kernel_inner=kernel_inner, kernel_outer=kernel_outer,
            wL=ring_wL, wS=ring_wS, wB=ring_wB,
            use_adaptive=use_adaptive, th_block=th_block, th_offset=th_offset
        )

        # Visualize channels and intermediate results (overlay GT)
        H = (res['H']*255).astype(np.uint8)
        L = (res['L']*255).astype(np.uint8)
        S = (res['S']*255).astype(np.uint8)
        R = (res['R']*255).astype(np.uint8)
        ringL = (res['ring_L']*255).astype(np.uint8)
        ringM = (res['ring_mix']*255).astype(np.uint8)
        m1 = (res['mask_raw']*255).astype(np.uint8)
        m2 = (res['mask_final']*255).astype(np.uint8)

        def _cm(gray):
            return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)

        vis_H = draw_points(_cm(H), gt_pts) if draw_gt else _cm(H)
        vis_L = draw_points(_cm(L), gt_pts) if draw_gt else _cm(L)
        vis_S = draw_points(_cm(S), gt_pts) if draw_gt else _cm(S)
        vis_R = draw_points(_cm(R), gt_pts) if draw_gt else _cm(R)
        vis_ringL = draw_points(_cm(ringL), gt_pts) if draw_gt else _cm(ringL)
        vis_ringM = draw_points(_cm(ringM), gt_pts) if draw_gt else _cm(ringM)
        vis_m1 = draw_points(cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR), gt_pts) if draw_gt else cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
        vis_m2 = draw_points(cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR), gt_pts) if draw_gt else cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)

        det_overlay = bgr.copy()
        for (x,y,w,h) in res['boxes']:
            cv2.rectangle(det_overlay,(x,y),(x+w,y+h),(0,255,0),2)
        for (cx,cy) in res['centers']:
            cv2.circle(det_overlay,(int(cx),int(cy)),4,(0,255,255),-1, lineType=cv2.LINE_AA)
        if draw_gt and gt_pts:
            det_overlay = draw_points(det_overlay, gt_pts, color=(255,255,255))

        # Output directory (one subfolder per image for easier inspection)
        img_out_dir = os.path.join(out_dir, name)
        os.makedirs(img_out_dir, exist_ok=True)

        # Save single-image outputs
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_H.jpg"), vis_H)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_L.jpg"), vis_L)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_S.jpg"), vis_S)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_Rbase.jpg"), vis_R)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_ringL.jpg"), vis_ringL)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_ringMix.jpg"), vis_ringM)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_mask_raw.jpg"), vis_m1)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_mask_final.jpg"), vis_m2)
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_det.jpg"), det_overlay)

        # Montage for quick overview
        row1 = montage_horiz([vis_H, vis_L, vis_S], gap=10)
        row2 = montage_horiz([vis_ringL, vis_ringM, vis_R], gap=10)
        row3 = montage_horiz([vis_m1, vis_m2, det_overlay], gap=10)
        panel = np.vstack([row1, row2, row3])
        cv2.imwrite(os.path.join(img_out_dir, f"{name}_steps.jpg"), panel)

        # Evaluation
        tp, fp, fn, prec, rec = match_points_to_centers(gt_pts, res['centers'], radius=match_radius)
        results.append([name, len(gt_pts), len(res['centers']), tp, fp, fn, prec, rec])

    # Save CSV summary
    out_csv = os.path.join(out_dir, "summary.csv")
    with open(out_csv, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_count", "det_count", "TP", "FP", "FN", "precision", "recall"])
        w.writerows(results)
    print(f"[OK] Done. Results -> {out_csv}")

# ========== CLI ==========
def parse_args():
    p = argparse.ArgumentParser("Parasite counter with HLS + ring enhancement (Top-hat + Black-hat)")
    #p.add_argument("--images", type=str, required=True, help="directory with segmented images")
    #p.add_argument("--labels", type=str, required=True, help="directory with matching CSV labels (columns: point_idx,x,y)")
    #p.add_argument("--out", type=str, required=True, help="output root directory")

    # Geometric filtering & evaluation
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--max-area", type=int, default=100)
    p.add_argument("--min-ar", type=float, default=1.0)
    p.add_argument("--max-ar", type=float, default=10.0)
    p.add_argument("--match-radius", type=int, default=12)
    p.add_argument("--draw-gt", action="store_true", help="overlay GT points on visualizations")

    # Watershed
    p.add_argument("--watershed", action="store_true", help="enable watershed (off by default)")

    # Ring enhancement and weights
    p.add_argument("--kernel-inner", type=int, default=9, help="inner kernel for ring enhancement (approx short-side of object)")
    p.add_argument("--kernel-outer", type=int, default=25, help="outer kernel for ring enhancement (object + dark border)")
    p.add_argument("--ring-wL", type=float, default=0.8, help="weight for ring response from L channel")
    p.add_argument("--ring-wS", type=float, default=0.2, help="weight for ring response from S channel")
    p.add_argument("--ring-wB", type=float, default=0.0, help="weight for legacy R_base map")

    # Thresholding
    p.add_argument("--adaptive", action="store_true", help="use adaptive threshold (default on)")
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="disable adaptive threshold and use Otsu instead")
    p.add_argument("--th-block", type=int, default=61, help="adaptive threshold block size (odd)")
    p.add_argument("--th-offset", type=int, default=+2, help="adaptive threshold offset")

    # Compatibility (kept for potential later use)
    p.add_argument("--ts", type=float, default=0.5, help="upper bound for S (used by legacy condition)")
    p.add_argument("--tl", type=float, default=0.6, help="lower bound for L (used by legacy condition)")
    return p.parse_args()

def main():
    args = parse_args()
    images_dir = "processed_output/B2025-00086a/segment/imgs"
    csv_dir = "processed_output/B2025-00086a/segment/labels"
    out_dir = "processed_output/B2025-00086a/segment/results"
    run_pipeline(
        images_dir=images_dir,
        csv_dir=csv_dir,
        out_dir=out_dir,
        ts=args.ts, tl=args.tl,
        area_range=(args.min_area, args.max_area),
        ar_range=(args.min_ar, args.max_ar),
        match_radius=args.match_radius,
        draw_gt=args.draw_gt,
        do_watershed=args.watershed,
        kernel_inner=args.kernel_inner, kernel_outer=args.kernel_outer,
        ring_wL=args.ring_wL, ring_wS=args.ring_wS, ring_wB=args.ring_wB,
        use_adaptive=args.adaptive if args.adaptive is not None else True,
        th_block=args.th_block, th_offset=args.th_offset
    )
if __name__ == "__main__":
    main()