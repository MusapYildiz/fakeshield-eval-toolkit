#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# FakeShield Pipeline — V100 Sunucu (Temiz Versiyon)
#
# Kullanım:
#   bash run_fakeshield.sh single <görüntü.jpg>           — DTE + MFLM
#   bash run_fakeshield.sh dte    <görüntü.jpg>           — sadece DTE
#   bash run_fakeshield.sh batch  <klasör/> [çıktı.jsonl] — toplu analiz
#   bash run_fakeshield.sh csv    <sonuçlar.jsonl>         — JSONL → CSV
# ═══════════════════════════════════════════════════════════════════

# ── Konfigürasyon ──────────────────────────────────────────────────
BASE_DIR="${FAKESHIELD_DIR:-$HOME/FakeShield}"
WEIGHT_DIR="/workspace/FakeShield/weight/fakeshield-v1-22b"   # container içi path
OUTPUT_DIR="$BASE_DIR/results"
CACHE_DIR="$BASE_DIR/.cache"
GPU_ID=4
DTE_IMAGE="zhipeixu/dte-fdm:v1.0"
MFLM_IMAGE="zhipeixu/mflm:v1.0-mmcv"
# ──────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# Gerekli klasörler
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# ── Yardımcı: DTE çıktısındaki /input/ path'ini düzelt ─────────────
fix_dte_path() {
    local DTE_OUT="$1"
    local REAL_IMG_DIR="$2"

    [ -f "$DTE_OUT" ] || return
    [ -s "$DTE_OUT" ] || return   # boş dosyayı atla

    python3 -c "
import json, sys

with open('$DTE_OUT', 'r') as f:
    content = f.read().strip()

if not content:
    sys.exit(0)

lines = [l.strip() for l in content.split('\n') if l.strip()]
fixed = []
for line in lines:
    try:
        d = json.loads(line)
        img = d.get('image', '')
        if '/input/' in img:
            fname = img.split('/input/')[-1]
            d['image'] = '/workspace/FakeShield/playground/images/' + fname
        elif any(x in img for x in ['/home/', '/kullanici_yedek/']):
            fname = img.split('/')[-1]
            d['image'] = '/workspace/FakeShield/playground/images/' + fname
        fixed.append(json.dumps(d, ensure_ascii=False))
    except:
        fixed.append(line)

with open('$DTE_OUT', 'w') as f:
    f.write('\n'.join(fixed) + '\n')
" 2>/dev/null
}

# ── Yardımcı: Sonucu göster ────────────────────────────────────────
show_result() {
    local DTE_OUT="$1"
    local MFLM_OUT="$2"

    echo ""
    echo "── SONUÇ ──────────────────────────────"

    if [ -f "$DTE_OUT" ] && [ -s "$DTE_OUT" ]; then
        python3 -c "
import json
with open('$DTE_OUT') as f:
    lines = [l.strip() for l in f if l.strip()]
if lines:
    try:
        d = json.loads(lines[-1])
        print(d.get('outputs', '')[:600])
    except:
        print(lines[-1][:600])
" 2>/dev/null
    else
        echo "DTE-FDM çıktısı yok"
    fi

    if [ -n "$MFLM_OUT" ] && [ -d "$MFLM_OUT" ]; then
        echo ""
        echo "Maske: $MFLM_OUT/"
        ls "$MFLM_OUT/" 2>/dev/null
    fi
    echo "───────────────────────────────────────"
}

# ── Fonksiyon: DTE-FDM çalıştır ────────────────────────────────────
run_dte() {
    local IMAGE_PATH="$1"
    local DTE_OUT="$2"
    local IMG_DIR=$(dirname "$IMAGE_PATH")
    local IMG_NAME=$(basename "$IMAGE_PATH")

    # Output dosyasını önceden oluştur (izin sorunu için)
    touch "$DTE_OUT"
    chmod 666 "$DTE_OUT"

    docker run --gpus "device=$GPU_ID" --rm \
        -v "$BASE_DIR:/workspace/FakeShield" \
        -v "$IMG_DIR:/input:ro" \
        -e TRANSFORMERS_CACHE=/workspace/FakeShield/.cache \
        -e HF_HOME=/workspace/FakeShield/.cache \
        "$DTE_IMAGE" \
        bash -c "
            cd /workspace/FakeShield
            pip install -q transformers==4.37.2 2>/dev/null
            CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
                --model-path $WEIGHT_DIR/DTE-FDM \
                --DTG-path   $WEIGHT_DIR/DTG.pth \
                --image-path /input/$IMG_NAME \
                --output-path /workspace/FakeShield/results/$(basename $DTE_OUT)
        " 2>&1 | grep -v "^$" | grep -v "UserWarning\|FutureWarning\|warn"

    fix_dte_path "$DTE_OUT" "$IMG_DIR"
}

# ── Fonksiyon: MFLM çalıştır ───────────────────────────────────────
run_mflm() {
    local DTE_OUT="$1"
    local MFLM_OUT="$2"

    [ -f "$DTE_OUT" ] && [ -s "$DTE_OUT" ] || { warn "DTE çıktısı boş, MFLM atlandı"; return 1; }

    mkdir -p "$MFLM_OUT"
    chmod 777 "$MFLM_OUT"

    docker run --gpus "device=$GPU_ID" --rm \
        -v "$BASE_DIR:/workspace/FakeShield" \
        -e TRANSFORMERS_CACHE=/workspace/FakeShield/.cache \
        -e HF_HOME=/workspace/FakeShield/.cache \
        "$MFLM_IMAGE" \
        bash -c "
            cd /workspace/FakeShield
            pip install -q transformers==4.28.0 2>/dev/null
            CUDA_VISIBLE_DEVICES=0 python ./MFLM/cli_demo.py \
                --version $WEIGHT_DIR/MFLM \
                --DTE-FDM-output /workspace/FakeShield/results/$(basename $DTE_OUT) \
                --MFLM-output    /workspace/FakeShield/results/$(basename $MFLM_OUT)
        " 2>&1 | grep -E "Mask saved|Error|error|MFLM"
}

# ── Tek görüntü — DTE + MFLM ───────────────────────────────────────
run_single() {
    local IMAGE_PATH="$1"
    [ -f "$IMAGE_PATH" ] || err "Görüntü bulunamadı: $IMAGE_PATH"

    local BASENAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    local DTE_OUT="$OUTPUT_DIR/${BASENAME}_dte.jsonl"
    local MFLM_OUT="$OUTPUT_DIR/${BASENAME}_mflm"

    echo "════════════════════════════════════════"
    echo " Analiz: $(basename $IMAGE_PATH)"
    echo "════════════════════════════════════════"

    log "DTE-FDM çalışıyor..."
    run_dte "$IMAGE_PATH" "$DTE_OUT"

    if [ -s "$DTE_OUT" ]; then
        log "DTE-FDM tamamlandı → $DTE_OUT"
        log "MFLM çalışıyor..."
        run_mflm "$DTE_OUT" "$MFLM_OUT"
        log "MFLM tamamlandı → $MFLM_OUT"
    else
        warn "DTE-FDM çıktısı boş"
    fi

    show_result "$DTE_OUT" "$MFLM_OUT"
}

# ── Sadece DTE-FDM ─────────────────────────────────────────────────
run_dte_only() {
    local IMAGE_PATH="$1"
    [ -f "$IMAGE_PATH" ] || err "Görüntü bulunamadı: $IMAGE_PATH"

    local BASENAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    local DTE_OUT="$OUTPUT_DIR/${BASENAME}_dte.jsonl"

    log "DTE-FDM çalışıyor: $(basename $IMAGE_PATH)"
    run_dte "$IMAGE_PATH" "$DTE_OUT"

    if [ -s "$DTE_OUT" ]; then
        log "Tamamlandı → $DTE_OUT"
    else
        warn "Çıktı boş"
    fi

    show_result "$DTE_OUT" ""
}

# ── Toplu Analiz ───────────────────────────────────────────────────
run_batch() {
    local IMG_DIR="$1"
    local BATCH_OUT="${2:-$OUTPUT_DIR/batch_results.jsonl}"

    [ -d "$IMG_DIR" ] || err "Klasör bulunamadı: $IMG_DIR"

    mapfile -t IMAGES < <(find "$IMG_DIR" -maxdepth 2 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)

    local TOTAL=${#IMAGES[@]}
    [ "$TOTAL" -gt 0 ] || err "Görüntü bulunamadı: $IMG_DIR"

    echo "════════════════════════════════════════"
    echo " Toplu Analiz: $TOTAL görüntü"
    echo " Çıktı: $BATCH_OUT"
    echo "════════════════════════════════════════"

    > "$BATCH_OUT"

    local COUNT=0
    for IMG in "${IMAGES[@]}"; do
        COUNT=$((COUNT + 1))
        local BASENAME=$(basename "$IMG" | sed 's/\.[^.]*$//')
        local DTE_OUT="$OUTPUT_DIR/${BASENAME}_dte.jsonl"

        echo -n "[$COUNT/$TOTAL] $(basename $IMG) ... "

        run_dte "$IMG" "$DTE_OUT" > /dev/null 2>&1

        if [ -f "$DTE_OUT" ] && [ -s "$DTE_OUT" ]; then
            python3 -c "
import json
with open('$DTE_OUT') as f:
    lines = [l.strip() for l in f if l.strip()]
if lines:
    try:
        d = json.loads(lines[-1])
        d['image_file'] = '$(basename $IMG)'
        print(json.dumps(d, ensure_ascii=False))
    except:
        print(json.dumps({'image': '$IMG', 'raw': lines[-1]}))
" >> "$BATCH_OUT" 2>/dev/null
            echo "OK"
        else
            echo "FAILED"
            echo "{\"image\": \"$IMG\", \"error\": \"no_output\"}" >> "$BATCH_OUT"
        fi
    done

    echo ""
    log "Tamamlandı: $BATCH_OUT"

    python3 -c "
import json
results = []
with open('$BATCH_OUT') as f:
    for line in f:
        line = line.strip()
        if line:
            try: results.append(json.loads(line))
            except: pass
ok    = sum(1 for r in results if 'error' not in r)
fail  = sum(1 for r in results if 'error' in r)
print(f'Toplam: {len(results)} | Başarılı: {ok} | Başarısız: {fail}')
" 2>/dev/null
}

# ── JSONL → CSV ────────────────────────────────────────────────────
to_csv() {
    local JSONL="$1"
    local CSV="${JSONL%.jsonl}.csv"

    python3 -c "
import json, csv
rows = []
with open('$JSONL') as f:
    for line in f:
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except: pass
if not rows:
    print('Veri yok')
    exit()
fields = list(rows[0].keys())
with open('$CSV', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
print(f'CSV: $CSV ({len(rows)} satır)')
"
}

# ── Ana Giriş ──────────────────────────────────────────────────────
case "$1" in
    single) run_single   "$2" ;;
    dte)    run_dte_only "$2" ;;
    batch)  run_batch    "$2" "$3" ;;
    csv)    to_csv       "$2" ;;
    *)
        echo "Kullanım:"
        echo "  $0 single <görüntü.jpg>             — DTE-FDM + MFLM (tam pipeline)"
        echo "  $0 dte    <görüntü.jpg>             — sadece DTE-FDM"
        echo "  $0 batch  <klasör/> [çıktı.jsonl]   — toplu analiz"
        echo "  $0 csv    <sonuçlar.jsonl>           — JSONL → CSV"
        echo ""
        echo "Örnek:"
        echo "  $0 single ~/FakeShield/playground/images/test.jpg"
        echo "  $0 batch  ~/test_images/ ~/FakeShield/results/batch.jsonl"
        ;;
esac
