#!/bin/bash
# CASIA GT test seti için toplu MFLM maskesi üretimi

SP_DIR="${FAKESHIELD_DIR:-$HOME/FakeShield}/datasets/casia_gt_test/Sp"
RESULTS_DIR="${FAKESHIELD_DIR:-$HOME/FakeShield}/results"
WEIGHT_DIR="/workspace/FakeShield/weight/fakeshield-v1-22b"

total=$(ls $SP_DIR/*.jpg | wc -l)
count=0

for img in $SP_DIR/*.jpg; do
    count=$((count + 1))
    basename=$(basename "$img" .jpg)
    dte_out="$RESULTS_DIR/${basename}_dte.jsonl"
    mflm_out="$RESULTS_DIR/${basename}_mflm"

    echo "[$count/$total] $basename"

    # DTE çıktısı yoksa atla
    if [ ! -f "$dte_out" ] || [ ! -s "$dte_out" ]; then
        echo "  ⚠️ DTE çıktısı yok, atlanıyor"
        continue
    fi

    # MFLM maskesi zaten varsa atla
    if [ -d "$mflm_out" ] && [ "$(ls $mflm_out 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  ✅ Maske zaten mevcut, atlanıyor"
        continue
    fi

    # Path düzelt
    python3 -c "
import json
with open('$dte_out') as f:
    lines = [l.strip() for l in f if l.strip()]
if lines:
    d = json.loads(lines[-1])
    img = d.get('image','')
    if '/Au/' in img or '/Sp/' in img:
        pass  # zaten doğru
    else:
        fname = img.split('/')[-1]
        d['image'] = '/workspace/FakeShield/datasets/casia_gt_test/Sp/' + fname
    with open('$dte_out', 'w') as f:
        json.dump(d, f)
        f.write('\n')
" 2>/dev/null

    # MFLM klasörünü hazırla
    mkdir -p "$mflm_out"
    chmod 777 "$mflm_out"

    # MFLM çalıştır
    docker run --gpus "device=6" --rm \
        -v "${FAKESHIELD_DIR:-$HOME/FakeShield}":/workspace/FakeShield \
        -v $SP_DIR:/workspace/FakeShield/datasets/casia_gt_test/Sp:ro \
        -e TRANSFORMERS_CACHE=/workspace/FakeShield/.cache \
        -e HF_HOME=/workspace/FakeShield/.cache \
        zhipeixu/mflm:v1.0-mmcv \
        bash -c "
            cd /workspace/FakeShield
            pip install -q transformers==4.28.0 2>/dev/null
            CUDA_VISIBLE_DEVICES=0 python ./MFLM/cli_demo.py \
                --version $WEIGHT_DIR/MFLM \
                --DTE-FDM-output /workspace/FakeShield/results/${basename}_dte.jsonl \
                --MFLM-output    /workspace/FakeShield/results/${basename}_mflm
        " 2>&1 | grep -E "Mask saved|Error|error"

done

echo ""
echo "Tamamlandı. Üretilen maskeler:"
ls $RESULTS_DIR | grep "_mflm$" | wc -l
