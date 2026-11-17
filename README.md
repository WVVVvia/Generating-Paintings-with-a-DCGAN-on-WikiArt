# ECE 285 HW2: DCGAN on WikiArt

A DCGAN trained on 20,000 WikiArt paintings at 64×64, evaluated with FID and Inception Score using pytorch-ignite.

![Uploading generated_samples.png…]()


## Results

10,000 samples, generator at epoch 80:

|                 | FID ↓ | IS ↑ |
|-----------------|------:|-----:|
| DCGAN           | 61.4  | 4.21 |
| Real images     | 5.8   | 7.58 |

The real-image FID is between two disjoint halves of the training set. Training took 41.9 minutes on one Tesla T4 (80 epochs, 12,480 iterations).

## Running it

The notebook `ece285_hw2_dcgan_wikiart.ipynb` is meant to run on Kaggle:

1. Import the notebook (File → Import Notebook).
2. Settings: Accelerator **GPU T4 x2**, Internet **On**.
3. Add Input: the `steubk/wikiart` dataset.
4. Run all. It takes about 50 minutes, including preprocessing.

All figures, `results.json`, and checkpoints are written to `/kaggle/working/outputs`. Settings such as the subset size, epochs, and learning rates are in the config cell at the top.

## Details

- **Data:** 20,000 random paintings from all 27 styles, center-cropped and resized to 64×64, with random horizontal flips.
- **Model:** standard DCGAN generator and discriminator (3.58M and 2.77M parameters), BCE loss with one-sided label smoothing (real label 0.9), Adam with lr 2e-4 and β = (0.5, 0.999), batch size 128.
- **Evaluation:** `ignite.metrics.FID` and `InceptionScore` on images upsampled to 299×299 and normalized with ImageNet statistics. FID uses the 2048-d Inception pool features; ignite's default computes it on the 1000-class softmax output instead.

## Files

```
ece285_hw2_dcgan_wikiart.ipynb   training and evaluation notebook
hw2_report.tex                   report (NeurIPS 2025 style)
outputs/                         figures and results.json from the run above
```
