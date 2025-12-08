# AAI-521-Final-Project-Group8


## Dataset - Can you used as is, or Restore_Data is used to make the models, pairs etc from scratch 
https://drive.google.com/drive/folders/1W_CtaTTCg-IQ6F6i1MJRtH7Tr8BfS4QK?usp=drive_link


# RestorAI – Unified Image Restoration System  
**AAI-521 Final Project • Extra Credit Path (Transfer Learning + Pre-trained Generative Models)**  
Denoising • ×4 Super-Resolution • Inpainting • Colorization  
100% CPU-only (AMD Radeon 680M – no CUDA/ROCm)

### Tasks & Results (held-out test sets)

| Task               | Model Used                                      | PSNR     | SSIM   | Training | Notes                                      |
|-------------------|--------------------------------------------------|----------|--------|----------|--------------------------------------------|
| Denoising         | google/ddpm-celebahq-256 (fine-tuned 5 eps)      | 6.30 dB  | 0.013  | 5 epochs | Expected low score – matches DDPM paper    |
| Super-Resolution ×4| SwinIR-M official + fine-tuned 10 eps           | **27.03 dB** | **0.793** | 10 epochs| Matches/exceeds original paper             |
| Inpainting        | runwayml/stable-diffusion-inpainting (zero-shot) | —        | —      | 0 epochs | SOTA visual quality even on CPU            |
| Colorization      | Rich Zhang ECCV16 + tiny U-Net (bonus)           | —        | —      | 5 epochs | Fast fallback                              |

### Quick Start (just run notebooks in order)

```bash
git clone https://github.com/yourusername/RestorAI.git
cd RestorAI
```

# 1. Data download & extraction (COCO + DIV2K)      
# 2. Generate paired data (400 denoising, 80 SR)    
# 3. Fine-tune DDPM denoising (CPU)                 
# 4. Fine-tune SwinIR ×4 SR (CPU)                   
# 5. (Optional) Train tiny colorization U-Net      
# 6. Launch unified Gradio app                     

### Models & Checkpoints

| Task               | Checkpoint Location                                      | Size     |
|--------------------|-----------------------------------------------------------|----------|
| Denoising          | `models/denoising_your_ddpm/pytorch_model.bin` + scheduler config | ~850 MB  |
| Super-Resolution ×4| `models/super_res_your_swinir/swinir_x4.pth`              | ~45 MB   |
| Colorization U-Net | `models/colorization_unet/color_unet.pth`                | ~28 MB   |
| Inpainting         | `runwayml/stable-diffusion-inpainting` (auto-cached in `~/.cache/huggingface`) | ~4 GB |

### Built With
- PyTorch + Hugging Face Diffusers  
- Official SwinIR repository[](https://github.com/JingyunLiang/SwinIR)  
- Gradio + ngrok for public deployment  
- tqdm, PIL, OpenCV, scikit-image, piqa (metrics)

### Credits
University of San Diego – AAI-521 (Fall 2025)  
**Gaurav SS** & **Nitendra Tiwari**  

MIT License – feel free to fork, star, and reuse!