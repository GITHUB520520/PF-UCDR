# PF-UCDR: A Local-Aware RGB-Phase Fusion Network with Adaptive Prompts for Universal Cross-Domain Retrieval


**Abstract:** Universal Cross-Domain Retrieval (UCDR) aims to match semantically related images across domains and categories not seen during training. While vision-language pre-trained models offer strong global alignment, we are inspired by the observation that local structures, such as shapes, contours, and textures often remain stable across domains, and thus propose to model them explicitly at the patch level. We present PF-UCDR, a framework built upon frozen vision-language backbones that performs patch-wise fusion of RGB and phase representations. Central to our design is a Fusing Vision Encoder, which applies masked cross-attention to spatially aligned RGB and phase patches, enabling fine-grained integration of complementary appearance and structural cues. Additionally, we incorporate adaptive visual prompts that condition image encoding based on domain and class context. Local and global fusion modules aggregate these enriched features, and a two-stage training strategy progressively optimizes alignment and retrieval objectives. Experiments on standard UCDR benchmarks demonstrate that PF-UCDR significantly outperforms existing methods, validating the effectiveness of structure-aware local fusion grounded in multimodal pretraining.

<!-- Replace with your actual architecture diagram -->
<p align="center">
  <img src="./PF-UCDR.png" width="800" alt="PF-UCDR Architecture">
</p>
<p align="center">
  <em>Overall architecture of PF-UCDR, illustrating training (left) and inference (right) stages.</em>
</p>

---

## 🛠️ Requirements

```shell
cd ./PF-UCDR 
conda create --name PF_UCDR python=3.10
conda activate PF_UCDR
pip install -r requirements.txt
```

---

## 💾 Data Preparation

1.  **Download Datasets**📥:
    We use DomainNet, Sketchy, and TU-Berlin. Please download them manually.
    *   **DomainNet**: Official Site: [http://ai.bu.edu/DomainNet/](http://ai.bu.edu/DomainNet/)
    *   **Sketchy**: Download Link (from ProS repository): [Google Drive Link](https://drive.google.com/drive/folders/1IGmRIP826s_aGYq004vbJfyaMGsCmUeD?usp=sharing)
    *   **TU-Berlin**: Download Link (from ProS repository): [Google Drive Link](https://drive.google.com/drive/folders/1qTEe5DjGdh45UFfGa6Ofy_80yJRp7uI5?usp=sharing)
    
    * Links for Sketchy and TU-Berlin are provided courtesy of the ProS project: https://github.com/kaipengfang/ProS

2.  **Expected Directory Structure**📁:
    Please organize the downloaded datasets under a root `data/` directory (or as configured in your scripts) according to the structure expected by our dataloaders. This structure is similar to common UCDR benchmarks:
    ```
    ├── DomainNet
    │   ├── clipart 
    │   ├── clipart_test.txt 
    │   ├── clipart_train.txt
    │   ├── down.sh
    │   ├── infograph
    │   ├── infograph_test.txt
    │   ├── infograph_train.txt
    │   ├── painting
    │   ├── painting_test.txt
    │   ├── painting_train.txt
    │   ├── quickdraw
    │   ├── quickdraw_test.txt
    │   ├── quickdraw_train.txt
    │   ├── real
    │   ├── real_test.txt
    │   ├── real_train.txt
    │   ├── sketch
    │   ├── sketch_test.txt
    │   └── sketch_train.txt
    ├── Sketchy
    │   ├── extended_photo
    │   ├── photo
    │   ├── sketch
    │   └── zeroshot1
    └── TUBerlin
        ├── images
        └── sketches
    ```

---

## ⚙️ Run

Training and evaluation scripts are `src/alogs/PF-UCDR/trainer.py` and `src/alogs/PF-UCDR/test.py` respectively. All configurations can be set via command-line arguments or by modifying `src/options/options.py`.
Your main execution script is assumed to be `main.py` at the project root, which then calls the trainer/tester.

### 🚀 Training PF-UCDR

* `-hd` (holdout domain for UCDR/UdCDR on DomainNet)
* `-bs` (batch size), `-lr` (learning rate)
* `-tp_N_CTX` (text prompt length)
* `-prompt` (visual prompt length per group)
* phase augmentation parameters like `--hpf_range`, `--hpf_alpha`, `--aug_alpha` as needed.

**DomainNet (UCDR/UdCDR):**
```bash
# UCDR
python3 main.py -data DomainNet -hd sketch -sd quickdraw -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd quickdraw -sd sketch -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd clipart -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd painting -sd infograph -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
python3 main.py -data DomainNet -hd infograph -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0

# UdCDR
python3 main.py -data DomainNet -hd sketch -sd quickdraw -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0 -ucddr 1
python3 main.py -data DomainNet -hd quickdraw -sd sketch -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0 -ucddr 1
python3 main.py -data DomainNet -hd clipart -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0 -ucddr 1
python3 main.py -data DomainNet -hd painting -sd infograph -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0 -ucddr 1
python3 main.py -data DomainNet -hd infograph -sd painting -bs 50 -log 15 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -debug_mode 0 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0 -ucddr 1
```

**Sketchy (UcCDR):**
```bash
python3 main.py -data Sketchy -bs 50 -lr 0.0001 -tp_N_CTX 16 -prompt 1 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
```

**TU-Berlin (UcCDR):**
```bash
python3 main.py -data TUBerlin -bs 50 -lr 0.0001 -tp_N_CTX 16 -prompt 1 --hpf_range 50 --hpf_alpha 0.4 --aug_alpha 1.0
```

### 🧪 Testing PF-UCDR

* Ensure you have trained models. Modify `-weight` to point to your trained model checkpoint, and set evaluation-specific arguments like `-hd`, `-gallery_domain` for DomainNet.

**DomainNet (UCDR & UdCDR evaluation):**
```bash
# UCDR
python -m src.alogs.PF_UCDR.test -data DomainNet -hd sketch -sd quickdraw -tp_N_CTX 16 -debug_mode 0 --weight xxx
python -m src.alogs.PF_UCDR.test -data DomainNet -hd quickdraw -sd sketch -tp_N_CTX 16 -debug_mode 0 --weight xxx
python -m src.alogs.PF_UCDR.test -data DomainNet -hd clipart -sd painting -tp_N_CTX 16 -debug_mode 0 --weight xxx
python -m src.alogs.PF_UCDR.test -data DomainNet -hd painting -sd infograph -tp_N_CTX 16 -debug_mode 0 --weight xxx
python -m src.alogs.PF_UCDR.test -data DomainNet -hd infograph -sd painting -tp_N_CTX 16 -debug_mode 0 --weight xxx

# UdCDR
python -m src.alogs.PF_UCDR.test -data DomainNet -hd sketch -sd quickdraw -tp_N_CTX 16 -debug_mode 0 --weight xxx -ucddr 1
python -m src.alogs.PF_UCDR.test -data DomainNet -hd quickdraw -sd sketch -tp_N_CTX 16 -debug_mode 0 --weight xxx -ucddr 1
python -m src.alogs.PF_UCDR.test -data DomainNet -hd clipart -sd painting -tp_N_CTX 16 -debug_mode 0 --weight xxx -ucddr 1
python -m src.alogs.PF_UCDR.test -data DomainNet -hd painting -sd infograph -tp_N_CTX 16 -debug_mode 0 --weight xxx -ucddr 1
python -m src.alogs.PF_UCDR.test -data DomainNet -hd infograph -sd painting -tp_N_CTX 16 -debug_mode 0 --weight xxx -ucddr 1
```

**Sketchy (UcCDR evaluation):**
```bash
python3 main.py -test -data Sketchy -bs 50 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -weight xxx
```

**TU-Berlin (UcCDR evaluation):**
```bash
python3 main.py -test -data TUBerlin -bs 50 -lr 0.0001 -tp_N_CTX 16 -prompt 1 -weight xxx
```
---

## 📊 Results

Our PF-UCDR framework achieves state-of-the-art or highly competitive performance on standard UCDR benchmarks, demonstrating significant improvements in generalizing to unseen domains and categories. Key results on DomainNet for the UCDR task are summarized below:

| Query Domain | Method             | Unseen Gallery (mAP@200 / Prec@200) | Mixed Gallery (mAP@200 / Prec@200) |
|--------------|--------------------|--------------------------------------|------------------------------------|
| Sketch       | **PF-UCDR (Ours)** | **0.6628** / **0.6183**              | **0.6098** / **0.5724**            |
| Quickdraw    | **PF-UCDR (Ours)** | **0.2891** / **0.2668**              | **0.2399** / **0.2255**            |
| Painting     | **PF-UCDR (Ours)** | **0.7604** / **0.7065**              | **0.7250** / **0.6759**            |
| Infograph    | **PF-UCDR (Ours)** | **0.5832** / **0.5503**              | **0.5384** / **0.5110**            |
| Clipart      | **PF-UCDR (Ours)** | **0.7748** / **0.7299**              | **0.7402** / **0.7001**            |
| Average      | **PF-UCDR (Ours)** | **0.6141** / **0.5744**              | **0.5707** / **0.5370**            |

For detailed comparisons including UdCDR and UcCDR results, and performance against other baselines on all datasets, please refer to our main paper.

---

## 🙏 Acknowledgements

We would like to acknowledge the contributions of the following works and their authors for making their code publicly available, which were valuable references for our research:
*   **ProS**: [https://github.com/kaipengfang/ProS](https://github.com/kaipengfang/ProS)
*   **UCDR-Adapter**: [https://github.com/fine68/UCDR2024](https://github.com/fine68/UCDR2024)
---
