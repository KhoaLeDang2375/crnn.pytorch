# CRNN (MobileNetV3 Powered)

This software implements the Convolutional Recurrent Neural Network (CRNN) in PyTorch for image-based sequence recognition (scene text recognition/OCR), based on the paper *An end-to-end trainable neural network for image-based sequence recognition*.

## 🚀 Key Updates & Features
1. **Lightweight Backbone**: Replaced the original VGG architecture with `MobileNetV3` (Pre-trained), ensuring much faster training & inference speed, ideal for mobile or edge deployments while preserving required sequence lengths.
2. **Kaggle & Modern PyTorch Ready**: Completely removed the outdated and buggy `warp_ctc_pytorch`. It now uses native `torch.nn.CTCLoss`, ensuring 0 conflicts out of the box on modern PyTorch (≥1.7).
3. **Multi-Language Support (Japanese/Chinese etc.)**: Added tools to extract custom character dictionaries dynamically, breaking the barrier of the hard-coded English-only alphabets.

---

## 💻 Environment Setup
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
# Requirements include: lmdb, Pillow, torch>=1.7.0, torchvision>=0.8.0
```

---

## 🏃‍♂️ Training Workflow (End-to-End)

If you have your own data (e.g., Japanese text images) labeled in a text file (where lines are formatted like `path/to/img.png \t Your_Label`), follow these steps:

### Step 1: Extract Alphabet Dictionary
The model needs to know exactly what characters exist in your dataset (especially important for non-English languages). Run the alphabet extractor tools to scan your ground-truth text files:
```bash
python tool/extract_alphabet.py
```
*This will create a `dict.txt` file containing all unique characters located in your training labels.*

### Step 2: Convert Dataset to LMDB formats
CRNN relies on the **LMDB** format for lightning-fast disk I/O during training. Convert your raw images and `txt` labels into LMDB format:
```bash
python tool/create_lmdb_dataset.py
```
*This generates an `lmdb_dataset/` directory holding your `train` and `val` data ready for PyTorch.*

### Step 3: Run Training (Kaggle or Local)
You can directly run training via the provided bash script. If using Kaggle, open the `train.bash` file to change `TRAIN_ROOT` and `VAL_ROOT` pointing to your dataset appropriately.

```bash
bash train.bash
```

Alternatively, run the manual python command:
```bash
python train.py \
    --trainRoot lmdb_dataset/train \
    --valRoot lmdb_dataset/val \
    --batchSize 64 \
    --imgH 32 \
    --imgW 100 \
    --nepoch 25 \
    --adadelta \
    --keep_ratio \
    --cuda \
    --dict dict.txt \
    --expr_dir expr
```
**Flags definition**:
- `--dict`: Path to your custom dictionary (e.g., `dict.txt`), overriding the default English alphabet.
- `--keep_ratio`: Pads images instead of stretching them, preserving the original text aspect ratio.
- `--cuda`: Train strictly on GPU.
- `--adadelta`: Adadelta optimizer (highly recommended for CRNN).

Expected output format:
```text
[0/25][100/1000] Loss: 2.1345
```
Model checkpoints `.pth` will be dumped into the `--expr_dir` (default: `expr/`) after training intervals.

---

## 📚 Cite
```tex
@article{shi2016end,
  title={An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition},
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={39},
  number={11},
  pages={2298--2304},
  year={2016},
  publisher={IEEE}
}
```
