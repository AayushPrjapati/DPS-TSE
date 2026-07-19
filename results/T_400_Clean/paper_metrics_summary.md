# Clean Environment (T = 400) — Paper Metrics Summary

The table below shows the exact average performance metrics reported in **Table I** of the IEEE conference paper draft for the clean Libri2Mix test set under the $T = 400$ step configuration.

| Method | D/G Type | SI-SNR (dB) $\uparrow$ | ESTOI $\uparrow$ | DNSMOS $\uparrow$ | WER (%) $\downarrow$ | SIM $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Mixture** | -- | -0.235 | 0.550 | 3.006 | N/A | N/A |
| **TFGridNet** (Discriminative Baseline) | D | **12.341** | **0.865** | _3.209_ | _1.662_ | **0.981** |
| **Proposed (Our V1)** (Generative) | G | _12.111_ | _0.825_ | **3.549** | **1.512** | _0.978_ |

* **WER Reduction:** Under $T = 400$ steps, the Proposed V1 model achieves the best Word Error Rate (WER) of **1.512%**, yielding an absolute reduction of **0.150%** over the discriminative TFGridNet baseline.
* **Logs Reference:** The raw data used to calculate these averages can be found in [`running_averages.csv`](running_averages.csv) and [`final_results.csv`](final_results.csv) in this folder.
