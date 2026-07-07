#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# FakeShield Pipeline — V100 Sunucu Kurulum ve Çalıştırma Scripti
# Kullanım: bash setup_and_run.sh
# ═══════════════════════════════════════════════════════════════════

set -e  # Hata olursa dur

# ── Konfigürasyon ──────────────────────────────────────────────────
BASE_DIR="$HOME/FakeShield"
WEIGHT_DIR="$BASE_DIR/weight/fakeshield-v1-22b"
SAM_PATH="$BASE_DIR/weight/sam_vit_h_4b8939.pth"
GPU_ID=4          # Boş V100-32GB
DTE_IMAGE="zhipeixu/dte-fdm:v1.0"
MFLM_IMAGE="zhipeixu/mflm:v1.0"

# Renkli çıktı
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Adım 1: Ön Kontroller ─────────────────────────────────────────
echo "════════════════════════════════════════"
echo " FakeShield Pipeline — Ön Kontrol"
echo "════════════════════════════════════════"

# Docker imajları var mı?
docker image inspect $DTE_IMAGE  > /dev/null 2>&1 && log "DTE-FDM imajı mevcut" || err "DTE-FDM imajı yok: docker pull $DTE_IMAGE"
docker image inspect $MFLM_IMAGE > /dev/null 2>&1 && log "MFLM imajı mevcut"   || err "MFLM imajı yok: docker pull $MFLM_IMAGE"

# Model ağırlıkları var mı?
[ -d "$WEIGHT_DIR/DTE-FDM" ] && log "DTE-FDM ağırlıkları mevcut" || err "Ağırlıklar eksik: $WEIGHT_DIR/DTE-FDM"
[ -d "$WEIGHT_DIR/MFLM"    ] && log "MFLM ağırlıkları mevcut"    || err "Ağırlıklar eksik: $WEIGHT_DIR/MFLM"
[ -f "$WEIGHT_DIR/DTG.pth" ] && log "DTG ağırlığı mevcut"        || err "Ağırlık eksik: $WEIGHT_DIR/DTG.pth"
[ -f "$SAM_PATH"           ] && log "SAM ağırlığı mevcut"         || warn "SAM yok — MFLM çalışmaz: $SAM_PATH"

# GPU müsait mi?
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null)
log "GPU $GPU_ID boş VRAM: ${GPU_MEM} MiB"
[ "$GPU_MEM" -gt 20000 ] || warn "GPU $GPU_ID'de yeterli VRAM olmayabilir (${GPU_MEM} MiB)"

echo ""

# ── Adım 2: SAM İndir (yoksa) ─────────────────────────────────────
if [ ! -f "$SAM_PATH" ]; then
    warn "SAM indiriliyor (~2.4GB)..."
    wget -q --show-progress \
        https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth \
        -O "$SAM_PATH"
    log "SAM indirildi"
fi

echo "════════════════════════════════════════"
echo " Hazır. Kullanım için aşağıdaki"
echo " fonksiyonları çalıştırın."
echo "════════════════════════════════════════"
