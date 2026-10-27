## Workflow of implement imitation learning
This folder provide the code sample of imitation learning policy following the LeRobot framework. However, the implement here only focus on evaluate the policies on datasets and have not inferences in practical experiments. 

So, the pipeline can be separated into six steps: explore dataset, data preparation, model configuration, model training, model inference and plotting results. First, dataset should be nailed to the correct form making sure it is suitable for LeRobot framework. Second, hyperparameters of chosen model, for example ACT or Diffusion, should be selected before training with dataset. Model inferences the testset after training, in this way, the performance of the model are obtained by comparing the results of inference with groundtruth of the testset. After all, the results are depicted as charts and metric values, such as MSE and MAE.

## Pipeline Workflow

The machine learning pipeline is structured into six sequential steps tailored for the **LeRobot** framework:

```mermaid
flowchart TD
    %% Define styles for clarity
    classDef stepStyle fill:#f0f4f8,stroke:#3b82f6,stroke-width:2px,color:#1e293b,rx:8,ry:8;

    S1["1. Explore Dataset\n• Inspect data structure and features"] --> S2
    S2["2. Data Preparation\n• Format dataset for LeRobot compatibility"] --> S3
    S3["3. Model Configuration\n• Select hyperparameters\n• Choose model architecture (e.g., ACT, Diffusion)"] --> S4
    S4["4. Model Training\n• Train model on prepared dataset"] --> S5
    S5["5. Model Inference\n• Run inference on test set\n• Compare predictions with ground truth"] --> S6
    S6["6. Plotting Results\n• Generate charts and performance metrics\n• Evaluate via MSE, MAE, etc."]

    class S1,S2,S3,S4,S5,S6 stepStyle;
