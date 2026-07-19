# Clean Environment (T = 100) — Paper Metrics Summary

The table below shows the exact average performance metrics reported in **Table I** of the IEEE conference paper draft for the clean Libri2Mix test set under the $T = 100$ step configuration.

| Method | D/G Type | SI-SNR (dB) $\uparrow$ | ESTOI $\uparrow$ | DNSMOS $\uparrow$ | WER (%) $\downarrow$ | SIM $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Mixture** | -- | -0.128 | 0.538 | 2.939 | N/A | N/A |
| **TFGridNet** (Discriminative Baseline) | D | **12.450** | **0.877** | _3.201_ | **1.979** | **0.982** |
| **Proposed (Our V1)** (Generative) | G | _12.287_ | _0.834_ | **3.546** | _2.445_ | _0.978_ |

* **DNSMOS Boost:** The Proposed V1 model achieves a significant boost of **+0.345** in speech naturalness (DNSMOS) over the discriminative TFGridNet baseline, while maintaining high speaker similarity and intelligibility.
* **Logs Reference:** The raw data used to calculate these averages can be found in [`running_averages.csv`](running_averages.csv) and [`individual_results.csv`](individual_results.csv) in this folder.
