# FakeShield Evaluation Toolkit

FakeShield (ICLR 2025) modelinin görüntü sahteciliği tespiti (splicing, inpainting, AI-generated content) performansını değerlendirmek için geliştirilmiş otomasyon ve analiz araçları.

Bu repo, [orijinal FakeShield reposu](https://github.com/zhipeixu/FakeShield) üzerine inşa edilmiş bağımsız bir değerlendirme katmanıdır — orijinal framework kodu değil, ona ait pipeline otomasyonu, IoU/metrik hesaplama ve sonuç görselleştirme scriptlerini içerir.

## İçerik
- `run_fakeshield.sh` — Docker tabanlı DTE-FDM + MFLM pipeline çalıştırma otomasyonu (tekli, batch, CSV çıktı modları)
- `run_mflm_batch.sh` — MFLM için toplu segmentasyon çalıştırma
- `evaluate_fakeshield.py` — Model çıktılarının otomatik değerlendirilmesi
- `evaluate_iou.py` — Segmentasyon maskeleri için IoU metrik hesaplama
- `generate_viewer.py` — Sonuçları HTML üzerinde görselleştiren araç
- `playground/eval_jsonl.py` — JSONL formatındaki test/çıktı dosyalarını değerlendirme
- `results_sample/` — Örnek çıktı formatları

## Kurulum
\`\`\`bash
export FAKESHIELD_DIR=/path/to/your/FakeShield
bash fakeshield_setup.sh
\`\`\`

## Bulgular
FakeShield, splicing tespitinde güçlü performans gösterirken inpainting/object-removal ve AI-generated içerik tespitinde belirgin zayıflık sergiledi.

## Lisans
Bu repo yalnızca değerlendirme/otomasyon kodunu içerir. Orijinal FakeShield modeli ve ağırlıkları için [orijinal repo](https://github.com/zhipeixu/FakeShield) ve lisansına bakınız.
