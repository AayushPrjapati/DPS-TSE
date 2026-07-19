# Noisy Environment (T = 100) — Paper Metrics Summary

The table below shows the exact average performance metrics reported in **Table I** of the IEEE conference paper draft for the noisy Libri2Mix test set under the $T = 100$ step configuration.

| Method | D/G Type | SI-SNR (dB) $\uparrow$ | ESTOI $\uparrow$ | DNSMOS $\uparrow$ | WER (%) $\downarrow$ | SIM $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Noisy Mix** | -- | -1.390 | 0.418 | 2.126 | N/A | N/A |
| **TFGridNet** (Discriminative Baseline) | D | **9.828** | **0.768** | _3.104_ | **14.466** | _0.955_ |
| **Proposed (Our V1)** (Generative) | G | _9.762_ | _0.742_ | **3.321** | _16.369_ | **0.965** |

* **Speaker Identity Stabilization:** In noisy environments, the generative prior stabilizes the target speaker's vocal characteristics, improving WavLM speaker similarity (SIM) by **+0.010** (reaching **0.965**) compared to the discriminative baseline.
* **Logs Reference:** The raw data used to calculate these averages can be found in [`running_averages.csv`](running_averages.csv) and [`individual_results.csv`](individual_results.csv) in this folder.
