#### 📄 Dosya 2: `Tez_Akis_Plani.md`
*(Detaylandırılmış Yol Haritası)*

```markdown
# FAZ 3: META-RL İLE HATA TOLERANSI ÇALIŞMA PLANI (DETAYLI)
**Strateji:** 1. Prototip (MuJoCo - Mac/Ubuntu) -> 2. Final (IsaacLab - Ubuntu)

---

### 🧠 Alt Faz 3.1: Model-Free & Context-Based Meta-RL (RL²)
**Amaç:** Ajanın hafızasını (RNN) kullanarak değişen fiziksel parametreleri (Context) öğrenmesi.

#### 🟢 Adım 3.1.0: Prototip - Reacher (Damping Değişimi)
* **Ortam:** `Reacher-v4` (MuJoCo).
* **Görev:** Robot kolunun eklem sürtünmesi (damping) her epizotta rastgele değişir.
* **Varyasyon Testi (Side-by-Side):**
    1.  `PPO + LSTM`: Uzun süreli hafıza.
    2.  `PPO + GRU`: Daha basit ve hızlı hafıza.
    3.  `PPO + FrameStack`: RNN olmadan sadece son 4 kare (Baseline).
* **Çıktı:** Hangi hafıza hücresi sürtünme değişimine daha hızlı adapte oluyor?

#### 🟡 Adım 3.1.1: Orta Seviye - Inverted Pendulum (Kütle Değişimi)
* **Ortam:** `InvertedPendulum-v4`.
* **Görev:** Sarkacın ucundaki kütle ($m$) $0.5x$ ile $5.0x$ arasında değişir.
* **Yöntem:** Adım 3.1.0'ın şampiyonu (örn. GRU) burada kullanılır.
* **Analiz:** Latent Space ($h_t$) görselleştirmesi. Hafif ve ağır kütleler uzayda ayrışıyor mu?

#### 🔴 Adım 3.1.2: IsaacLab Final - Unitree A1 (Kaygan Zemin)
* **Platform:** Ubuntu (IsaacLab).
* **Senaryo:** Robot yürürken zemin sürtünmesi ($\mu$) aniden $1.0 \to 0.2$ (buz) seviyesine düşer.
* **Yöntem:** SKRL kütüphanesi içinde `SharedModel`'e entegre edilmiş, Adım 3.1.0'dan gelen şampiyon RNN hücresi.
* **Beklenen Sonuç:** Kaygan zeminde adımlarını kısaltarak dengede kalan robot (Hoca Demosu).

---

### ⚡ Alt Faz 3.2: Model-Free & Gradient-Based Meta-RL (MAML)
**Amaç:** Ajanın yeni bir arızaya 1-2 gradient adımında adapte olabilen "süper-başlangıç" ağırlıklarını öğrenmesi.

#### 🟢 Adım 3.2.0: En Basit Başlangıç - 2D Navigation
* **Ortam:** 2D Point Mass.
* **Görev:** Hedef nokta değişir, ajan 1-2 denemede hedefi bulmalıdır.
* **Varyasyon Testi:**
    1.  `MAML (Second Order)`: Tam türev hesabı.
    2.  `Reptile (First Order)`: Gradient yönüne gitme.

#### 🟡 Adım 3.2.1: Orta Seviye - Ant (Crippled Leg)
* **Ortam:** `Ant-v4`.
* **Görev:** Her epizotta rastgele bir bacak kilitlenir (Joint Locked).
* **Süreç:** Ajan 10 adım atar, düşer (gradient update), sonraki denemede topallayarak yürür.

#### 🔴 Adım 3.2.2: IsaacLab Final - Unitree A1 (Motor Degradasyonu)
* **Platform:** Ubuntu (IsaacLab).
* **Senaryo:** Sol arka motor torku %70 azalır.
* **Yöntem:** Önceden eğitilmiş politika, arıza anında alınan verilerle "Online Fine-Tuning" yapılır.

---

### 🔮 Alt Faz 3.3: Model-Based & Context-Based (Adaptive Dynamics)
**Amaç:** Ajan dünyanın fiziğini ($S' = f(S,A,z)$) öğrenir.

#### 🟢 Adım 3.3.0: Pendulum (Gravity)
* **Ortam:** `Pendulum-v1`. Yerçekimi değişir.
* **Varyasyon Testi:** `Deterministic Encoder` vs `Probabilistic Encoder (PE-TS)`.

#### 🟡 Adım 3.3.1: HalfCheetah (Fiziksel Değişim)
* **Ortam:** `HalfCheetah-v4`. Kütle ve sürtünme değişir.
* **Yöntem:** Kazanan Encoder + Dynamics Model + MPC.

#### 🔴 Adım 3.3.2: IsaacLab Final - Unitree A1 (Joint Stuck Estimation)
* **Senaryo:** Ön sağ diz kilitlenir.
* **Yöntem:** "Context Estimator Network". Robotun hareketine bakıp kilitli eklemi tahmin eder ve politikaya bildirir.

---

### 📈 Alt Faz 3.4: Model-Based & Gradient-Based (Online SysID)
**Amaç:** Fiziksel modelin anlık verilerle kendini eğitmesi.

#### 🟢 Adım 3.4.0: Reacher (Link Length)
* **Ortam:** `Reacher-v4`. Kol uzunluğu değişir.
* **Yöntem:** GrBAL (Gradient-Based Adaptive Learner).

#### 🔴 Adım 3.4.1: IsaacLab Final - Unitree A1 (Kombine Arıza)
* **Senaryo:** Bilinmeyen kombine arıza (Motor zayıf + Yük var).
* **Yöntem:** Saf PyTorch ile yazılmış, sürekli öğrenen Forward Dynamics Model ve Safety Filter.