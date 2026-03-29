# SanbornMap: Legend-Guided Zero-Annotation Segmentation  

## 1. Project Overview  

This project focuses on building material segmentation on Sanborn fire insurance maps, with strong robustness to color shift, aging, and scan artifacts.  
The long-term goal is a legend-guided segmentation model that matches map regions to standard legend semantics, reducing reliance on absolute color values.

---

## 2. Current Status  
The repository currently implements the front stages of the pipeline:

1. **Baseline (`baseline.ipynb`)**  
   Fixed-color matching for Brick/Wood + connected component filtering + IoU evaluation.

2. **Synthetic Data Generator (`generate_synthetic_data.py`)**  
   Creates map-like synthetic training data with:
   - polygon buildings,
   - text/line occlusions,
   - historical degradation simulation (noise, stains, vignette, global darkening).

3. **Pseudo/Synth2Real Training (`gen_pseudo.ipynb`)**  
   UNet training with synthetic data and validation on real maps.  
   Best checkpoint is selected by real validation performance.

4. **Config Standardization**  
   Added `config.example.json`, unified key naming (use `labels_dir`).


---

##  Planned Full Method

The full planned model is a legend-guided architecture with four core modules:

* **Legend Encoder (Support Branch)**: Inputs legend patches (64×64) for each class (e.g., Brick/Wood) and outputs class embeddings $F_{legend}$.
* **Map Encoder (Query Branch)**: Inputs map patches (512×512) and outputs dense feature maps $F_{map}$.
* **Cross-Attention Matching Module**: Uses map features as the query ($Q$) and legend features as the key/value ($K/V$) to compute dynamic matching:
  $$F_{attn} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Decoder & Skip Fusion**: Fuses the attention features $F_{attn}$ with intermediate map encoder features via skip connections, decoding them into final prediction logits.

---

## Next Steps  

- Implement `LegendEncoder` and dynamic legend patch loader.  
- Integrate cross-attention module into the segmentation backbone. 
- Filter high-confidence pseudo-labels from the baseline U-Net.
- Train the new architecture using the existing pseudo-label pipeline.  
---

## Repository Structure  

```text
.
├── baseline.ipynb                # fixed-color baseline 
├── gen_pseudo.ipynb              # synth2real training 
├── generate_synthetic_data.py    # synthetic data generation 
├── config.example.json           # example config
└──  requirements.txt              # dependencies 
```

---

## Quick Start  
1. Copy config template:
   `cp config.example.json`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run baseline and training notebooks in order.


---

## 7. Notes  
This repo currently emphasizes reproducible pipeline development and model iteration.  
Final legend-guided cross-attention implementation is in progress.


