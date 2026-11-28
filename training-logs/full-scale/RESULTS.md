# Results so far

The measurements below are held-out results. The named papers motivate the experiments and each study compares its implementation against a baseline on the same data split. Complete results, trained models, split IDs and the training commands are available in the linked local folders and the adjacent trainer README files.

| Professor | Real-data benchmark | Held-out result |
| --- | --- | --- |
| Guy Brock | 68,579 cells from the 10x PBMC expression dataset | Stability-based selection increased agreement under simulated gene dropout from 0.241 to 0.468 ARI. The selected clustering had lower silhouette, 0.123 versus 0.254. This is a within-donor comparison. |
| Jeff Leek | 1,937 METABRIC samples and five held-out cohorts | The raw MLP scored 0.940 macro AUROC. The residual and control-feature corrections scored 0.924 and 0.925. In the aligned simulation, both corrections failed in the cohort with the extreme association reversal. |
| Qingyun Wu | 44,064 eligible chronological Electricity records after timestamp cleanup | Successive fidelity used 32% less search CPU time and raised test average precision from 0.899 to 0.906 in this run. The aligned simulation showed a small average precision decrease. |
| René Vidal | Official Fashion-MNIST split, 60,000 training and 10,000 test images | Sparse OMP increased clean test ARI from 0.396 to 0.414 versus cosine neighbors. After noise and occlusion, ARI decreased from 0.187 to 0.180. |
| James Zou | 581,012 UCI Covertype records | Data-OOB achieved 0.798 average precision for identifying injected label flips versus 0.121 for random ranking. On the original holdout labels, pruning with Data-OOB improved log loss from 0.430 to 0.312 versus using all noisy training labels. The class-normalized variant ranked injected errors less accurately on this dataset. |
| Suhang Wang | 24,492 Amazon Ratings graph nodes | Hard class grouping increased macro-F1 from 0.404 to 0.430 across three splits versus mean aggregation. Its log loss increased from 1.363 to 1.536. Soft grouping scored 0.428 macro-F1 and 1.372 log loss. |
| Qunhua Li | 72,811 candidate intervals from independent ENCODE CTCF measurements | The empirical-null mixture increased average precision from 0.841 to 0.894 versus the fixed-null mixture. The geometric-mean rank baseline performed better at 0.941. |
| Paul Bogdan | Subject-disjoint EEG and WISDM forecasting | On WISDM, fractional features reduced forecast RMSE from 3.457 to 3.436 versus a long-history MLP. On EEG, RMSE decreased from 33.369 to 33.258 µV. This model had slightly higher error at the longest EEG horizon. |
| Jihun Hamm | 13,910 UCI gas sensor measurements spanning ten chronological batches | The frozen source scored 0.482 macro-F1, above anchored adaptation (0.467) and unguarded entropy adaptation (0.455). Anchoring reduced the degradation from entropy adaptation and its log loss was closer to the frozen model. |
| Polina Golland | 1,875 BUS-BRA scans with patient-disjoint train, validation and test sets | Conditioning on the loss weights reduced boundary Brier score from 0.1912 to 0.1890. Test Dice was 0.8105 versus 0.8119, with a paired 95% interval of −0.0109 to 0.0080 for the difference. |
| Mehak Gupta | 12,000 PhysioNet ICU records | The monotone GRU increased macro average precision from 0.793 to 0.796 against the independent-window GRU. It produced zero decreasing risks across increasing forecast windows, compared with 0.014% for the independent GRU. |
| Alex K. Jones | Two published Twitter trace prefixes, 600,000 requests each | At the smallest simulated capacity, learned admission reduced the traffic proxy from 102.77 to 84.58 weighted bytes per request versus LRU, and miss rate from 30.0% to 28.3%. At larger capacities it kept a lower traffic proxy while second-touch admission could attain a lower miss rate. These are replay measurements. |
| Huixin Zhan | Measured appliance demand from 136 complete days, replayed with simulated battery dynamics | With preference 0.55, the adaptive training rule reduced the held-out charge-cost proxy from 3.546 to 3.471 per day, while battery throughput rose from 0.500 to 0.551 kWh. The time-of-use heuristic had a lower cost proxy at 3.293 and used 2.716 kWh of battery throughput. |

See the [training index](README.md) for the implementation map and run [the verifier](verify.py) to check both studies and their saved artifacts for each professor.

The [final 15 professor pairs](remaining/RESULTS.md) have a separate comparison table.
