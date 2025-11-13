# AAI-521-Final-Project-Group8


## All are done only once, then shared between the teammates
### Shared drive content
```
from google.colab import drive
drive.mount('/content/drive')

# Create folders
!mkdir -p "/content/drive/MyDrive/RestorAI_Data/raw"
!mkdir -p "/content/drive/MyDrive/RestorAI_Data/paired"
!mkdir -p "/content/drive/MyDrive/RestorAI_Data/models"
```

### Dataset - public 
```
# === COCO 2017 Val (for Denoising, Colorization, Inpainting) ===
!wget -q http://images.cocodataset.org/zips/val2017.zip -O /content/drive/MyDrive/RestorAI_Data/raw/coco_val.zip
!unzip -q /content/drive/MyDrive/RestorAI_Data/raw/coco_val.zip -d /content/drive/MyDrive/RestorAI_Data/raw/coco_val/

# === DIV2K (for Super-Resolution) ===
!wget -q https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -O /content/drive/MyDrive/RestorAI_Data/raw/DIV2K_train.zip
!unzip -q /content/drive/MyDrive/RestorAI_Data/raw/DIV2K_train.zip -d /content/drive/MyDrive/RestorAI_Data/raw/DIV2K/

print("ALL DATASETS DOWNLOADED (NO ERRORS)")
```

### Generating Paired Data 
```
import os, glob, numpy as np
from PIL import Image
from tqdm import tqdm

DATA_PATH = "/content/drive/MyDrive/RestorAI_Data"
RAW = f"{DATA_PATH}/raw"
PAIRED = f"{DATA_PATH}/paired"

os.makedirs(f"{PAIRED}/denoising/clean", exist_ok=True)
os.makedirs(f"{PAIRED}/denoising/noisy", exist_ok=True)
os.makedirs(f"{PAIRED}/super_res/hr", exist_ok=True)
os.makedirs(f"{PAIRED}/super_res/lr", exist_ok=True)

# === DENOISING: COCO Val → Noisy (500 images) ===
coco_paths = sorted(glob.glob(f"{RAW}/coco_val/val2017/*.jpg"))[:500]
np.random.seed(42)  # Deterministic
for i, path in tqdm(enumerate(coco_paths), total=len(coco_paths)):
    img = Image.open(path).convert("RGB").resize((256, 256))
    img.save(f"{PAIRED}/denoising/clean/{i:04d}.png")
    arr = np.array(img) / 255.0
    noise = np.random.randn(*arr.shape) * 0.15
    noisy = np.clip(arr + noise, 0, 1)
    Image.fromarray((noisy * 255).astype('uint8')).save(f"{PAIRED}/denoising/noisy/{i:04d}.png")

# === SUPER-RES: DIV2K → LR x4 (100 images) ===
div2k_paths = sorted(glob.glob(f"{RAW}/DIV2K/DIV2K_train_HR/*.png"))[:100]
for i, path in tqdm(enumerate(div2k_paths), total=len(div2k_paths)):
    hr = Image.open(path).convert("RGB").resize((512, 512))
    lr = hr.resize((128, 128), Image.BICUBIC)
    hr.save(f"{PAIRED}/super_res/hr/{i:04d}.png")
    lr.save(f"{PAIRED}/super_res/lr/{i:04d}.png")

print("PAIRED DATA READY!")
```