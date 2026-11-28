# Remaining professor studies

Each trainer fits two independent studies, one for the paper-inspired method and one companion with a public measured dataset. Several companions impose a synthetic stress condition on measured inputs. Training and metric details are in the saved results JSON for each experiment. The associated model files and held-out arrays are kept alongside it.

| Professor | Trainer and method | Public-data evaluation | Research |
| --- | --- | --- | --- |
| Jia Li | [Spatial image annotation](spatial_tags.py) | 60,000 CIFAR-100 images, official split. Coarse-category tagging with global and spatial descriptors. | [Source](https://pubmed.ncbi.nlm.nih.gov/18421105/) |
| Ehsan Azimi | [Anisotropic calibration](overlay.py) | TUD-L intrinsics and reference poses. Marker observations and calibration errors are simulated. | [Source](https://arxiv.org/abs/1703.05834) |
| Aggelos Katsaggelos | [Learned reconstruction prior](plugplay.py) | MSD hippocampus MRI slices with artificial cosine coefficient masking. Test volumes are disjoint. | [Source](https://openaccess.thecvf.com/content/ICCV2021W/LCI/papers/Ekmekci_What_Does_Your_Computational_Imaging_Algorithm_Not_Know_A_Plug-and-Play_ICCVW_2021_paper.pdf) |
| Jian Peng | [Multiplex diffusion](network_diffusion.py) | Original human network and cellular-component annotation bundle. Transductive graph edges are visible while test gene annotations are held out. | [Source](https://cb.csail.mit.edu/mashup/) |
| Juan Cui | [Cross-omics recovery](cross_omics.py) | 177 matched primary tumors in TCGA-PAAD via UCSC Xena. A cross-omics association task with held-out cases. | [Source](https://digitalcommons.unl.edu/computerscidiss/205/) |
| Liqing Zhang | [Semi-supervised assay adaptation](assay_transfer.py) | UCI TCGA Pan-Cancer expression. Artificial assay shifts, with distinct source, adaptation and test records. | [Source](https://research.cs.vt.edu/zhanglab/publication.html) |
| Yuri Pritykin | [Tumor immune states](tumor_states.py) | GEO GSE120575 human melanoma baseline immune-cell profiles. Small patient cohort and exploratory out-of-fold evaluation. | [Source](https://pritykinlab.github.io/publication/2021-cd8t/) |
| Linwei Wang | [Missing ECG lead morphology](lead_recovery.py) | Patient-separated PTB-XL+ GE 12SL records. Recovers tabulated lead descriptors. | [Source](https://www.rit.edu/cblwang/learning-disentangled-representations) |
| Daniel Moyer | [Device-invariant representation](scanner_invariance.py) | BUS-BRA ultrasound domain-shift companion. Holds out the most frequent scanner and all its patients. | [Source](https://arxiv.org/abs/1904.05375) |
| Ajita Rattani | [EMG time-frequency classification](gesture.py) | UCI EMG gestures. Subject-disjoint holdout and six gesture categories. | [Source](https://vcbsl.github.io/lab-website/publications.html) |
| Jun Bai | [Region graph tissue recognition](tissue_graph.py) | PathMNIST official splits. Graph connections are between regions within a real tissue patch. Tissue classes supply patch-level supervision. | [Source](https://arxiv.org/abs/2212.12844) |
| Amir A. Amini | [PSF-aware motion correction](motion_field.py) | MSD Heart MRI slices. Displacement and anisotropic blur are simulated and evaluation volumes held out. | [Source](https://cdn.aaai.org/Symposia/Spring/1994/SS-94-05/SS94-05-022.pdf) |
| Xiaobai Liu | [Contextual trajectory decoding](trajectory.py) | UCI HAR official person-disjoint split. A sequence-labeling companion on inertial windows. | [Source](https://pubmed.ncbi.nlm.nih.gov/24136437/) |
| David Dickensheets | [Focal-plane classification](focus.py) | BBBC006 published microscope image-quality features. A well-disjoint measurement of focus prediction. | [Source](https://opg.optica.org/ol/abstract.cfm?uri=ol-42-7-1241) |
| Sandeep Singhal | [Clinical and expression fusion](oncology_fusion.py) | METABRIC age, pathology and expression with an entire cohort held out for 5-year outcome evaluation. | [Source](https://campus.und.edu/directory/sandeep.singhal) |

Run any trainer with the Python environment under real-data-projects/kvasir-robust-segmentation/.venv. Scripts use root-relative data paths. The [requirements file](requirements.txt) records the package versions used. Each script runs its aligned and public-data modes. The [verifier](../verify.py) counts a study after the result and model artifacts exist.

The [held-out comparison table](RESULTS.md) links the metrics for both projects per professor.
