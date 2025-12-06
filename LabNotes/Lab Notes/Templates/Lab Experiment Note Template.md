---
type: experiment
date:
  "{ time }":
phase: 3.1.0
status: 🟡 Running / 🟢 Success / 🔴 Fail / ⚪ Discarded
run_id: BURAYA_KODDAN_GELEN_ID_YAZILACAK
tags:
  - experiment
---

# [[{{date}}]] - Deney Başlığı (Örn: LSTM Layer Sayısını 2'ye Çıkarma)

## 1. Hipotez ve Amaç
**Neden bu deneyi yapıyorum?**
* Önceki deneyde (`[[Link-to-Previous-Exp]]`) loss düşmesine rağmen reward artmıyordu.
* *Hipotez:* Modelin hafıza kapasitesi (memory capacity) yetersiz kalıyor olabilir. LSTM katman sayısını 1'den 2'ye çıkarmak, karmaşık damping değişimlerini daha iyi modelleyebilir.

## 2. Değişiklikler 
**Baseline'dan farkı ne?**
* `configs/phase_3_1/reacher_lstm.yaml`:
    * `n_lstm_layers`: `1` -> `2`
    * `learning_rate`: `3e-4` (Sabit)
* **Git Commit:** `a1b2c3d` (Code Snapshot)

## 3. Sonuçlar 
*(Deney bittikten sonra burayı doldur)*

### Grafikler
![[20251125_12418]] 
*(Buraya Tensorboard'dan "Mean Reward" ve "Value Loss" ekran görüntüsünü yapıştır)*

### Metrikler
| Metrik | Değer |
| :--- | :--- |
| **Mean Reward** | -5.42 |
| **Adaptation Steps** | 150 |
| **Training Time** | 45 dk |

## 4. Analiz ve Yorum 
**Ne öğrendik?**
* 2 katmanlı LSTM, eğitimi yavaşlattı (duvar saati süresi %30 arttı).
* Reward eğrisinde belirgin bir iyileşme yok. Hatta `overfitting` belirtileri var (Train reward artıyor, Eval reward sabit).
* **Karar:** Sorun hafıza kapasitesi değil, muhtemelen `entropy coefficient` çok düşük, ajan keşfetmiyor.

## 5. Sonraki Adım 
- [ ] Bu konfigürasyonu iptal et (Discard).
- [ ] `ent_coef` değerini 0.0'dan 0.01'e artırarak yeni deney başlat. -> [[EXP-20241125-3.1.0-Reacher-HighEntropy]]