# ACT Policy Online RL Training Guide

## Sorun Neydi?

Önceki `Act_RL.py` scripti **sadece evaluation (test) için** tasarlanmıştı - hiç training kodu yoktu! Model kaydetme, optimizer, loss computation gibi training bileşenleri hiç yoktu.

## Çözüm

İki ayrı script oluşturuldu:

1. **`train_Act_RL.py`** - ACT policy'yi online RL ile EĞİTMEK için ✅
2. **`Act_RL.py`** - Eğitilmiş modelleri TEST ETMEK için (değiştirilmedi) ✅

---

## 🚀 Training (Eğitim) Nasıl Yapılır?

### Basit Kullanım (State-only, kamera yok)

```bash
cd /home/beable/IsaacLab-SO_100
python scripts/skrl/train_Act_RL.py \
    --task Template-So-100-FishRod-CubeLift-v0 \
    --num_envs 4 \
    --max_iterations 5000 \
    --chunk_size 10 \
    --save_interval 100
```

### Vision-based Training (Kameralarla)

```bash
python scripts/skrl/train_Act_RL.py \
    --task Template-So-100-FishRod-CubeLift-v0 \
    --num_envs 4 \
    --max_iterations 5000 \
    --chunk_size 10 \
    --use_vision \
    --image_size 128 \
    --save_interval 100
```

### Checkpoint'ten Devam Etme

```bash
python scripts/skrl/train_Act_RL.py \
    --task Template-So-100-FishRod-CubeLift-v0 \
    --resume outputs/act_rl_checkpoints/latest.pt \
    --num_envs 4 \
    --max_iterations 10000
```

---

## 📊 Training Çıktıları

Training sırasında her 10 iterasyonda metrikler yazdırılır:

```
[ITER 100/5000]
  Time: 45.23s
  Episodes: 124
  Avg Reward: 15.234
  Avg Length: 450.2
  Success Rate: 23.4%
  Policy Loss: 0.1234
  Value Loss: 0.5678
  Entropy: 0.0234
```

Checkpoint'ler şu dizine kaydedilir:
- `outputs/act_rl_checkpoints/checkpoint_iter_100.pt`
- `outputs/act_rl_checkpoints/checkpoint_iter_200.pt`
- `outputs/act_rl_checkpoints/latest.pt` (her zaman en son)

---

## 🎮 Evaluation (Test) Nasıl Yapılır?

Eğitim tamamlandıktan sonra, modeli test etmek için **orijinal Act_RL.py** scriptini kullan:

```bash
python scripts/skrl/Act_RL.py \
    --checkpoint outputs/act_rl_checkpoints/latest.pt \
    --task Template-So-100-FishRod-CubeLift-v0 \
    --num_rollouts 10 \
    --horizon 1000
```

---

## ⚙️ Önemli Parametreler

### Training Hiperparametreleri

| Parametre | Açıklama | Default |
|-----------|----------|---------|
| `--num_envs` | Paralel environment sayısı | 4 |
| `--max_iterations` | Toplam training iterasyonu | 5000 |
| `--horizon` | Episode başına max step | 1000 |
| `--learning_rate` | Learning rate | 3e-4 |
| `--gamma` | Discount factor | 0.99 |
| `--gae_lambda` | GAE lambda | 0.95 |
| `--clip_epsilon` | PPO clip epsilon | 0.2 |
| `--entropy_coef` | Entropy coefficient | 0.001 |
| `--ppo_epochs` | PPO update epoch sayısı | 4 |
| `--batch_size` | Mini-batch size | 256 |

### ACT Model Parametreleri

| Parametre | Açıklama | Default |
|-----------|----------|---------|
| `--chunk_size` | Action chunk size | 10 |
| `--hidden_dim` | Hidden layer boyutu | 256 |
| `--num_layers` | Transformer layer sayısı | 4 |
| `--num_heads` | Attention head sayısı | 8 |
| `--use_vision` | Vision encoder kullan | False |
| `--image_size` | Image boyutu (vision için) | 128 |

### Checkpoint Parametreleri

| Parametre | Açıklama | Default |
|-----------|----------|---------|
| `--save_interval` | Kaç iterasyonda bir kaydet | 100 |
| `--log_interval` | Kaç iterasyonda bir log | 10 |
| `--checkpoint_dir` | Checkpoint dizini | outputs/act_rl_checkpoints |
| `--resume` | Checkpoint'ten devam et | None |

---

## 🏗️ Architecture (Mimari)

### ACTPolicy (Actor Network)

```
Input: State (6D joint positions) + Optional Image (128x128x6)
  ↓
State Encoder (MLP: 6 → 256)
Vision Encoder (CNN: 6x128x128 → 128) [opsiyonel]
  ↓
Fusion Layer (concat + MLP: 256+128 → 256)
  ↓
Transformer Encoder (4 layers, 8 heads)
  ↓
Action Decoder (MLP: 256 → chunk_size * 6)
  ↓
Output: Action Chunk (chunk_size, 6) + Log Std
```

### ValueNetwork (Critic Network)

```
Input: State (6D) + Optional Image (128x128x6)
  ↓
State Encoder (MLP: 6 → 256)
Vision Encoder (CNN: 6x128x128 → 128) [opsiyonel]
  ↓
Fusion (concat + MLP: 256+128 → 256)
  ↓
Value Head (MLP: 256 → 128 → 64 → 1)
  ↓
Output: State Value (scalar)
```

---

## 📈 Training Algorithm (PPO)

1. **Rollout Collection**: Her iteration'da `horizon` step kadar veri topla
2. **GAE Computation**: Advantage ve return hesapla
3. **PPO Updates**: `ppo_epochs` kez mini-batch update yap
   - Policy loss (clipped objective)
   - Value loss (MSE)
   - Entropy bonus
4. **Checkpoint Save**: Her `save_interval` iterasyonda kaydet

---

## 🔍 Troubleshooting

### Problem: Training başlamıyor

**Çözüm**: Task isminin doğru olduğundan emin ol:
```bash
python scripts/list_envs.py  # Mevcut task'ları listele
```

### Problem: Out of Memory (OOM)

**Çözüm**: Batch size veya num_envs azalt:
```bash
python scripts/skrl/train_Act_RL.py --num_envs 2 --batch_size 128
```

### Problem: Reward artmıyor

**Çözüm**: Hyperparameter'ları ayarla:
```bash
python scripts/skrl/train_Act_RL.py \
    --learning_rate 1e-4 \
    --entropy_coef 0.01 \
    --clip_epsilon 0.3
```

### Problem: Checkpoint yüklenmiyor

**Çözüm**: Dosya yolunu kontrol et:
```bash
ls -lh outputs/act_rl_checkpoints/
python scripts/skrl/train_Act_RL.py --resume outputs/act_rl_checkpoints/latest.pt
```

---

## 📚 Dosya Yapısı

```
IsaacLab-SO_100/
├── scripts/skrl/
│   ├── train_Act_RL.py          # ✅ YENİ: Training scripti
│   ├── Act_RL.py                # Evaluation scripti (değiştirilmedi)
│   └── train_bc_optimized.py    # BC training (önceden vardı)
├── outputs/
│   └── act_rl_checkpoints/      # ✅ YENİ: Checkpoint dizini
│       ├── checkpoint_iter_100.pt
│       ├── checkpoint_iter_200.pt
│       └── latest.pt
└── ACT_RL_TRAINING_GUIDE.md     # ✅ YENİ: Bu dosya
```

---

## 🎯 Sonraki Adımlar

1. **State-only training ile başla** (hızlı debug için):
   ```bash
   python scripts/skrl/train_Act_RL.py --num_envs 4 --max_iterations 1000
   ```

2. **İyi sonuç alınca vision ekle**:
   ```bash
   python scripts/skrl/train_Act_RL.py --use_vision --image_size 128 --max_iterations 5000
   ```

3. **Hyperparameter tuning yap**:
   - Learning rate azalt/arttır
   - Chunk size ayarla
   - Entropy coefficient değiştir

4. **Best checkpoint'i evaluate et**:
   ```bash
   python scripts/skrl/Act_RL.py --checkpoint outputs/act_rl_checkpoints/checkpoint_iter_1000.pt --num_rollouts 20
   ```

---

## ✅ Özet

| Özellik | Durum |
|---------|-------|
| ✅ ACT Policy Architecture | Eklendi |
| ✅ Value Network (Critic) | Eklendi |
| ✅ PPO Training Loop | Eklendi |
| ✅ GAE Advantage Estimation | Eklendi |
| ✅ Rollout Buffer | Eklendi |
| ✅ Checkpoint Saving | Eklendi |
| ✅ Training Metrics | Eklendi |
| ✅ Vision Support | Eklendi |
| ✅ Resume from Checkpoint | Eklendi |
| ✅ Dual Camera Support | Eklendi |

**Artık ACT policy'yi online RL ile eğitebilirsin!** 🎉








