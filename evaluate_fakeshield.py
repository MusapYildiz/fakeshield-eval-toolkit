#!/usr/bin/env python3
import os
"""
FakeShield Değerlendirme Scripti
Kullanım: python3 evaluate.py
"""

import os, json, re
from pathlib import Path
from collections import Counter

# ── Konfigürasyon ──────────────────────────────────────────────────
BASE_DIR       = os.environ.get("FAKESHIELD_DIR", "./FakeShield")
RESULTS_DIR    = f"{BASE_DIR}/results"
GROUND_TRUTH   = f"{BASE_DIR}/datasets/saakshi_test/ground_truth.json"
REPORT_OUT     = f"{RESULTS_DIR}/evaluation_report.json"
# ──────────────────────────────────────────────────────────────────

def parse_verdict(output_text):
    """DTE-FDM çıktısından fake/real kararı çıkar."""
    if not output_text:
        return "unknown"

    text = output_text.lower()

    # "has been tampered" veya "has not been tampered" gibi ifadeler
    if any(p in text for p in [
        "has been tampered",
        "has been manipulated",
        "is a deepfake",
        "is fake",
        "is ai-generated",
        "is artificially generated",
        "is synthetic",
        "tampered area",
        "forgery",
        "manipulated",
        "inserted",
        "cut-and-paste",
        "fabricated"
    ]):
        return "fake"

    if any(p in text for p in [
        "has not been tampered",
        "is not tampered",
        "is authentic",
        "is genuine",
        "is real",
        "no signs of manipulation",
        "no evidence of tampering",
        "appears to be real",
        "no manipulation"
    ]):
        return "real"

    # "1. Whether..." formatında cevap varsa ilk satıra bak
    lines = output_text.strip().split('\n')
    for line in lines[:5]:
        line_lower = line.lower()
        if "has been tampered" in line_lower or "tampered" in line_lower:
            return "fake"
        if "has not been tampered" in line_lower or "not tampered" in line_lower:
            return "real"

    return "unknown"


def explanation_keywords(output_text):
    """Açıklamada geçen artifact kategorilerini çıkar."""
    if not output_text:
        return []

    text = output_text.lower()
    categories = {
        "lighting":    ["lighting", "light source", "illumination", "highlight", "shadow"],
        "edges":       ["edge", "blend", "seamless", "sharp", "boundary"],
        "resolution":  ["resolution", "sharpness", "blur", "focus", "pixelat"],
        "texture":     ["texture", "skin tone", "skin texture", "surface"],
        "shadows":     ["shadow", "cast shadow"],
        "perspective": ["perspective", "angle", "size", "proportion", "orientation"],
        "color":       ["color", "colour", "tone", "hue"],
        "physics":     ["physical law", "natural", "unnatural", "defies"],
    }

    found = []
    for cat, keywords in categories.items():
        if any(kw in text for kw in keywords):
            found.append(cat)
    return found


def load_dte_result(sample_file):
    """Bir örnek için DTE-FDM çıktısını yükle."""
    stem = Path(sample_file).stem  # örn: sample_001_label1
    dte_path = f"{RESULTS_DIR}/{stem}_dte.jsonl"

    if not os.path.exists(dte_path):
        return None
    if os.path.getsize(dte_path) == 0:
        return None

    with open(dte_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except:
        return {"outputs": lines[-1]}


def main():
    # Ground truth yükle
    with open(GROUND_TRUTH) as f:
        ground_truth = json.load(f)

    print(f"Ground truth: {len(ground_truth)} örnek")
    print(f"Fake: {sum(1 for g in ground_truth if g['label']==1)} | "
          f"Real: {sum(1 for g in ground_truth if g['label']==0)}")
    print()

    # Her örnek için karşılaştır
    results = []
    not_found = 0

    for gt in ground_truth:
        dte = load_dte_result(gt['file'])

        if dte is None:
            not_found += 1
            results.append({
                "file":          gt['file'],
                "gt_label":      gt['label_str'],
                "gt_label_int":  gt['label'],
                "fs_verdict":    "not_analyzed",
                "correct":       None,
                "gt_explanation": gt['explanation'][:200],
                "fs_explanation": None,
                "fs_artifacts":  [],
                "gt_confidence": gt['confidence']
            })
            continue

        fs_output  = dte.get('outputs', '')
        fs_verdict = parse_verdict(fs_output)
        gt_label   = gt['label_str']  # "fake" veya "real"

        # Doğru mu?
        correct = None
        if fs_verdict != "unknown":
            correct = (fs_verdict == gt_label)

        results.append({
            "file":           gt['file'],
            "gt_label":       gt_label,
            "gt_label_int":   gt['label'],
            "fs_verdict":     fs_verdict,
            "correct":        correct,
            "gt_explanation": gt['explanation'][:300],
            "fs_explanation": fs_output[:300] if fs_output else None,
            "fs_artifacts":   explanation_keywords(fs_output),
            "gt_confidence":  gt['confidence']
        })

    # ── Metrikler ──────────────────────────────────────────────────
    analyzed    = [r for r in results if r['fs_verdict'] != "not_analyzed"]
    decided     = [r for r in analyzed if r['fs_verdict'] != "unknown"]
    correct     = [r for r in decided if r['correct'] == True]

    fake_gt     = [r for r in results if r['gt_label'] == 'fake']
    real_gt     = [r for r in results if r['gt_label'] == 'real']

    tp = sum(1 for r in fake_gt if r['fs_verdict'] == 'fake')   # fake → fake
    fn = sum(1 for r in fake_gt if r['fs_verdict'] == 'real')   # fake → real
    fp = sum(1 for r in real_gt if r['fs_verdict'] == 'fake')   # real → fake
    tn = sum(1 for r in real_gt if r['fs_verdict'] == 'real')   # real → real

    accuracy  = len(correct) / len(decided) if decided else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Artifact dağılımı
    all_artifacts = []
    for r in analyzed:
        all_artifacts.extend(r['fs_artifacts'])
    artifact_dist = Counter(all_artifacts)

    # ── Rapor ──────────────────────────────────────────────────────
    print("=" * 55)
    print("  FakeShield Değerlendirme Raporu")
    print("  Dataset: saakshigupta/deepfake-detection-dataset-v3")
    print("=" * 55)
    print(f"\nToplam örnek    : {len(results)}")
    print(f"Analiz edilen   : {len(analyzed)}")
    print(f"Analiz edilmemiş: {not_found}")
    print(f"Karar verilenler: {len(decided)}")
    print(f"Belirsiz (unknown): {len(analyzed) - len(decided)}")

    print(f"\n── Confusion Matrix ────────────────")
    print(f"  TP (fake→fake) : {tp:3d}")
    print(f"  TN (real→real) : {tn:3d}")
    print(f"  FP (real→fake) : {fp:3d}  ← False Alarm")
    print(f"  FN (fake→real) : {fn:3d}  ← Miss")

    print(f"\n── Metrikler ───────────────────────")
    print(f"  Accuracy  : {accuracy:.3f}  ({len(correct)}/{len(decided)})")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")

    print(f"\n── Artifact Kategorileri ───────────")
    for art, count in artifact_dist.most_common():
        bar = "█" * count
        print(f"  {art:<12}: {count:3d} {bar}")

    print(f"\n── Yanlış Tahminler ────────────────")
    wrong = [r for r in decided if not r['correct']]
    for r in wrong:
        print(f"  {r['file']}")
        print(f"    GT: {r['gt_label'].upper():4s} | FakeShield: {r['fs_verdict'].upper()}")
        if r['fs_explanation']:
            print(f"    FS: {r['fs_explanation'][:150]}")
        print()

    print(f"\n── Belirsiz Tahminler ──────────────")
    unknown = [r for r in analyzed if r['fs_verdict'] == 'unknown']
    for r in unknown:
        print(f"  {r['file']} (GT: {r['gt_label']})")

    # JSON rapor kaydet
    report = {
        "dataset":    "saakshigupta/deepfake-detection-dataset-v3",
        "model":      "FakeShield DTE-FDM",
        "total":      len(results),
        "analyzed":   len(analyzed),
        "metrics": {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        },
        "artifact_distribution": dict(artifact_dist.most_common()),
        "per_sample": results
    }

    with open(REPORT_OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Rapor kaydedildi: {REPORT_OUT}")
    print("=" * 55)


if __name__ == "__main__":
    main()
