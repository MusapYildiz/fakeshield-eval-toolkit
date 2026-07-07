#!/usr/bin/env python3
import os
"""
FakeShield MFLM Mask IoU Değerlendirme Scripti
Kullanım: python3 evaluate_iou.py

MFLM maskelerini CASIA ground truth maskeleriyle karşılaştırır.
Metriks: IoU, Pixel-F1, Precision, Recall
"""

import os, json
import numpy as np
from PIL import Image
from pathlib import Path

# ── Konfigürasyon ──────────────────────────────────────────────────
BASE_DIR    = os.environ.get("FAKESHIELD_DIR", "./FakeShield")
GT_DIR      = f"{BASE_DIR}/datasets/casia_gt_test/GT"
SP_DIR      = f"{BASE_DIR}/datasets/casia_gt_test/Sp"
RESULTS_DIR = f"{BASE_DIR}/results"
REPORT_OUT  = f"{RESULTS_DIR}/iou_report.json"
# ──────────────────────────────────────────────────────────────────


def load_mask_binary(path, threshold=127):
    """Maskeyi binary numpy array olarak yükle."""
    img = Image.open(path).convert('L')  # Grayscale
    arr = np.array(img)
    return (arr > threshold).astype(np.uint8)


def resize_to_match(mask, target_shape):
    """Maskeyi hedef boyuta yeniden boyutlandır."""
    img = Image.fromarray(mask * 255)
    img = img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def compute_iou(pred, gt):
    """IoU (Intersection over Union) hesapla."""
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_pixel_metrics(pred, gt):
    """Pixel-level precision, recall, F1 hesapla."""
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
        'tp': int(tp), 'fp': int(fp),
        'fn': int(fn), 'tn': int(tn)
    }


def find_mflm_mask(img_name):
    """Bir görüntü için MFLM mask dosyasını bul."""
    stem = img_name.replace('.jpg', '').replace('.png', '')
    mflm_dir = f"{RESULTS_DIR}/{stem}_mflm"

    if not os.path.exists(mflm_dir):
        return None

    # Klasör içindeki ilk görüntüyü bul
    for f in os.listdir(mflm_dir):
        if f.endswith(('.jpg', '.png')):
            return f"{mflm_dir}/{f}"
    return None


def find_gt_mask(img_name):
    """Bir görüntü için ground truth maskesini bul."""
    stem = img_name.replace('.jpg', '').replace('.png', '')
    gt_path = f"{GT_DIR}/{stem}_gt.png"
    return gt_path if os.path.exists(gt_path) else None


def main():
    # Sp görüntülerini listele
    sp_images = sorted([f for f in os.listdir(SP_DIR)
                        if f.endswith(('.jpg', '.png'))])

    print(f"Toplam Sp görüntü: {len(sp_images)}")
    print()

    results = []
    iou_scores = []
    f1_scores  = []
    no_mask    = []
    no_gt      = []

    for img_name in sp_images:
        mflm_path = find_mflm_mask(img_name)
        gt_path   = find_gt_mask(img_name)

        if mflm_path is None:
            no_mask.append(img_name)
            results.append({'image': img_name, 'status': 'no_mflm_mask'})
            continue

        if gt_path is None:
            no_gt.append(img_name)
            results.append({'image': img_name, 'status': 'no_gt_mask'})
            continue

        # Maskeleri yükle
        try:
            pred_mask = load_mask_binary(mflm_path)
            gt_mask   = load_mask_binary(gt_path)

            # Boyut eşleştir
            if pred_mask.shape != gt_mask.shape:
                pred_mask = resize_to_match(pred_mask, gt_mask.shape)

            # Metrikler
            iou     = compute_iou(pred_mask, gt_mask)
            metrics = compute_pixel_metrics(pred_mask, gt_mask)

            iou_scores.append(iou)
            f1_scores.append(metrics['f1'])

            status = '✅' if iou > 0.3 else '⚠️' if iou > 0.1 else '❌'
            print(f"{status} {img_name[:50]}")
            print(f"   IoU: {iou:.3f} | F1: {metrics['f1']:.3f} | "
                  f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f}")

            results.append({
                'image':     img_name,
                'status':    'ok',
                'iou':       round(iou, 4),
                'f1':        round(metrics['f1'], 4),
                'precision': round(metrics['precision'], 4),
                'recall':    round(metrics['recall'], 4),
                'mflm_path': mflm_path,
                'gt_path':   gt_path
            })

        except Exception as e:
            print(f"❌ HATA: {img_name}: {e}")
            results.append({'image': img_name, 'status': f'error: {e}'})

    # ── Özet ──────────────────────────────────────────────────────
    ok_results = [r for r in results if r.get('status') == 'ok']

    print()
    print("=" * 60)
    print("  MFLM Lokalizasyon Değerlendirmesi")
    print("  Dataset: CASIA1 (Ground Truth Maskeli)")
    print("=" * 60)
    print(f"Toplam Sp görüntü    : {len(sp_images)}")
    print(f"MFLM maskesi bulunan : {len(ok_results)}")
    print(f"MFLM maskesi eksik   : {len(no_mask)}")
    print(f"GT maskesi eksik     : {len(no_gt)}")
    print()

    if iou_scores:
        print(f"── IoU Metrikleri ──────────────────────────")
        print(f"  Mean IoU  : {np.mean(iou_scores):.4f}")
        print(f"  Median IoU: {np.median(iou_scores):.4f}")
        print(f"  Max IoU   : {np.max(iou_scores):.4f}")
        print(f"  Min IoU   : {np.min(iou_scores):.4f}")
        print()
        print(f"── Pixel F1 Metrikleri ─────────────────────")
        print(f"  Mean F1   : {np.mean(f1_scores):.4f}")
        print(f"  Median F1 : {np.median(f1_scores):.4f}")
        print()
        print(f"── IoU Dağılımı ────────────────────────────")
        print(f"  IoU > 0.5 (iyi)    : {sum(1 for s in iou_scores if s > 0.5)}")
        print(f"  IoU 0.3-0.5 (orta) : {sum(1 for s in iou_scores if 0.3 <= s <= 0.5)}")
        print(f"  IoU 0.1-0.3 (zayıf): {sum(1 for s in iou_scores if 0.1 <= s < 0.3)}")
        print(f"  IoU < 0.1 (kötü)   : {sum(1 for s in iou_scores if s < 0.1)}")

    # JSON rapor
    report = {
        "dataset":   "CASIA1",
        "model":     "FakeShield MFLM",
        "total_sp":  len(sp_images),
        "evaluated": len(ok_results),
        "metrics": {
            "mean_iou":    round(float(np.mean(iou_scores)), 4) if iou_scores else None,
            "median_iou":  round(float(np.median(iou_scores)), 4) if iou_scores else None,
            "mean_f1":     round(float(np.mean(f1_scores)), 4) if f1_scores else None,
            "median_f1":   round(float(np.median(f1_scores)), 4) if f1_scores else None,
        },
        "per_sample": results
    }

    with open(REPORT_OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Rapor kaydedildi: {REPORT_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
