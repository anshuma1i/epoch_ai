## Evaluation

Evaluation Metric

Submissions are evaluated using Mean (macro-averaged) Average Precision (mAP) over all nine classes. The final score is the arithmetic mean of the Average Precision (AP) calculated for each of the 9 columns (Clutter + 8 Bird Species).

The final score is calculated as:
![Evaluation metric](eval_score.png)
Where is the Average Precision for class . The nine classes are:

    Clutter
    Cormorants
    Pigeons
    Ducks
    Geese
    Gulls
    Birds of Prey
    Waders
    Songbirds

Note: For Average Precision we use the SK-learn implementation.

A notebook with the implementation of the metric can be found in the following link.
Submission Format

Submissions in this competition must be done by uploading a single CSV or Parquet file. Each row corresponds to a unique radar track_id. The required columns are:

    track_id: The unique identifier for the track.
    9 Class Columns: A predicted probability (float between 0.0 and 1.0) for each of the 9 classes listed above.

For an exact template, refer to dataset/sample_submission.csv.