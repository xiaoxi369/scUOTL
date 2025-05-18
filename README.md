# scUOTL: Integrating single-cell RNA-seq and ATAC-seq data with transfer learning and unbalanced optimal transport
# Abstract
Integrating single-cell multi-omics data, particularly scRNA-seq and scATAC-seq, is crucial for revealing cellular heterogeneity and regulatory mechanisms. However, this task remains technically challenging due to heterogeneous feature spaces across modalities, the frequent absence of paired data, and imbalanced distributions between modalities. To address these challenges, we propose scUOTL, a semi-supervised integration framework that combines Unbalanced Optimal Transport (UOT) with transfer learning to efficiently align scRNA-seq and scATAC-seq data. scUOTL employs a shared neural encoder to project data from different modalities into a unified latent space and incorporates a dual alignment strategy in both the embedding and label spaces, thereby reduces modality-specific discrepancies and enabling accurate label transfer. The UOT mechanism flexibly models asymmetric cross-modal distributions, enhancing robustness in the presence of limited annotations or noisy labels. Systematic evaluations on five real-world datasets demonstrate that scUOTL consistently outperforms existing methods in integration quality, label transfer accuracy, and scalability. Overall, scUOTL offers a robust and biologically meaningful solution for integrating unpaired and imbalanced single-cell multi-omics data.
# Overview
![](https://github.com/xiaoxi369/scUOTL/blob/main/figures/scUOTL.png)
# System requirements
- Python >= 3.6
- numpy >= 1.26.4
- pandas >= 2.2.2
- scikit-learn >= 1.5.1
- scanpy >= 1.10.2
- POT >= 0.9.4
- seaborn >= 0.13.2
- torch >= 2.3.1
# Download code
Clone the repository with
 <pre>git clone https://github.com/xiaoxi369/scUOTL.git</pre>
# Running scUOTL
In terminal, run
 <pre>python src/main.py</pre> 
