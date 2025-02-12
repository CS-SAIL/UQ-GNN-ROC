# UQ-GNN-ROC

The Python implementation of CR-ROC (**Conditional Prediction ROC Bands for Graph Classification**) was published @ AISTATS-2025. [Arxiv](https://arxiv.org/abs/2410.15239)

## Method
Graph classification in medical imaging and drug discovery requires accuracy and robust uncertainty quantification. To address this need, we introduce Conditional Prediction ROC (CP-ROC) bands, offering uncertainty quantification for ROC curves and robustness to distributional shifts in test data. Although developed for Tensorized Graph Neural Networks (TGNNs), CP-ROC is adaptable to general Graph Neural Networks (GNNs) and other machine learning models. We establish statistically guaranteed coverage for CP-ROC under {\it a local exchangeability condition}. This addresses uncertainty challenges for ROC curves under non-iid setting, ensuring reliability when test graph distributions differ from training data. Empirically, to establish local exchangeability for TGNNs, we introduce a data-driven approach to construct local calibration sets for graphs. Comprehensive evaluations show that CP-ROC significantly improves prediction reliability across diverse tasks. This method enhances uncertainty quantification efficiency and reliability for ROC curves, proving valuable for real-world applications with non-iid objects.

## Requirements
Python 3.10, torch 2.0.0, gudhi 3.7.1, tensorly-torch 0.4.0, networkx 3.0, numpy 1.24.2, scipy 1.10.1, scikit-learn 1.2.2.

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation

```{r}
@misc{wu2024conditionalpredictionrocbands,
      title={Conditional Prediction ROC Bands for Graph Classification}, 
      author={Yujia Wu and Bo Yang and Elynn Chen and Yuzhou Chen and Zheshi Zheng},
      year={2024},
      eprint={2410.15239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.15239}, 
}
