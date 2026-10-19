# End-to-End MLOps for Heart Disease Prediction
**Course:** MLOps AIMLCZG523  
**Assignment:** 01  
**Student:** [Your name]  
**Date:** [Submission date]  
**Repository:** [Your GitHub URL]

> Complete this template using your own outputs and explanations. Do not leave placeholders or invent results.

## 1. Abstract
Summarize the problem, data, models, MLOps workflow, actual results and limitations.

## 2. Problem statement and objectives
Explain the classification task and why reproducibility, testing, packaging, deployment and monitoring matter.

## 3. Dataset acquisition and understanding
- Dataset source and access method:
- Number of rows and columns from your downloaded file:
- Target definition:
- Feature descriptions and coding:
- Missing-value findings:

## 4. Exploratory data analysis
Insert your own class balance plot, feature histograms and correlation heatmap. Describe patterns you actually observe; avoid claiming correlation proves causation.

## 5. Data preparation and feature engineering
Explain the train/test split, imputation, numeric scaling, categorical encoding, and how you avoid data leakage by fitting preprocessing within the pipeline.

## 6. Model development and evaluation
Describe Logistic Regression and Random Forest, cross-validation, random seed, and model-selection metric. Insert your actual `artifacts/metrics.json` results in a table.

| Model | CV accuracy | CV precision | CV recall | CV ROC-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [actual] | [actual] | [actual] | [actual] | [actual] |
| Random Forest | [actual] | [actual] | [actual] | [actual] | [actual] |

Discuss trade-offs and why the selected model was chosen.

## 7. Experiment tracking with MLflow
Explain the experiment name, logged parameters/metrics/artifacts, and add a screenshot of your actual MLflow UI.

## 8. Packaging and reproducibility
Explain the saved pipeline, requirements file, setup commands, and how preprocessing is bundled with the estimator.

## 9. API, Docker and local verification
Document `/health` and `/predict`, input schema, response format, Docker build/run commands, and screenshots from your own successful tests.

## 10. CI/CD and automated tests
Explain the tests, lint step, dataset acquisition, training job, artifact upload, and include a screenshot of a successful GitHub Actions run. Include any failures and how you fixed them.

## 11. Deployment and monitoring
Explain whether you used Minikube, Docker Desktop Kubernetes, or a public cloud. Include the actual deployment manifest, service access instructions, pod/service status, request logs, and monitoring evidence. Do not claim a public URL unless it works.

## 12. Architecture
Include a diagram showing dataset → EDA/preprocessing → model training → MLflow → saved model → FastAPI → Docker → Kubernetes/cloud, with GitHub Actions for CI and logs/metrics for monitoring.

## 13. Challenges, limitations and ethical considerations
Discuss data size, missing values, class imbalance, generalization, possible bias, dataset coding, and why this educational model must not be used for clinical decisions.

## 14. Conclusion and future improvements
Summarize what you implemented and propose improvements such as external validation, drift monitoring, security, model registry, and more robust deployment.

## References
Cite the UCI dataset, library documentation, and any other sources you actually used.

## Appendix
Include commands, test output, selected screenshots, and the short demo video link or file reference.
