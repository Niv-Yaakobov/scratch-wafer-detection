
# Scratch Detection on Semiconductor Wafers

This project builds a machine learning model to detect scratches on semiconductor wafer dies based on their position and neighborhood features.

## Problem Statement

In semiconductor manufacturing, it is important to detect scratched dies while minimizing unnecessary inspections. However, on wafers with a low overall yield (i.e., many bad dies), scratch detection is skipped because the entire wafer will be manually inspected anyway. The task is to predict scratch locations on high-yield wafers without leaking information from labels that would not be available at test time.

## Approach

- **Data Cleaning**: Converted label columns to integers for clean aggregation operations.
- **Wafer Yield Calculation**: Computed wafer yields to identify low-yield wafers.
- **Neighbor Features**: Created `GoodNeighbors` and `FaultyNeighbors` features based only on the `IsGoodDie` label, to simulate real test conditions where scratch labels are unknown.
- **Training Set Construction**: Trained only on wafers with normal yield (>92%). Created a balanced dataset by sampling more good dies to balance rare scratch samples.
- **Model**: Trained a Random Forest classifier with 50 trees.
- **Threshold Tuning**: Applied a probability threshold (0.7) to optimize recall for scratched dies while keeping false positives acceptable.
- **Test Prediction**: Predicted on all dies, as yield information was not available at prediction time.

## Folder Structure

```
project/
│
├── scratch_detection_assignment.ipynb   # Main solution notebook
├── submission.csv                        # Final test predictions
└── README.txt                             # This file
```

## Key Considerations

- No use of `IsScratchDie` labels for neighbor feature creation.
- Low-yield wafers are excluded from training but included in prediction.
- Clean separation of training, validation, and test pipelines.
- Fully reproducible (random seeds set).

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- (optional) matplotlib or seaborn for any exploration

Install requirements with:

```bash
pip install -r requirements.txt
```

(You can generate a `requirements.txt` later with `pip freeze > requirements.txt`.)

## Final Notes

This project was completed as part of a professional assignment.  
Future improvements could include:
- Dynamic threshold adjustment based on wafer yield.
- Specialized models for low-yield wafers.
- Semi-supervised learning for uncertain cases.

---

> Created by Niv Cohen
