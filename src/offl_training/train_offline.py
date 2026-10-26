"""
Offline training script
"""
import numpy as np
from src.offl_training.classifier import OnlineLDA
from src.offl_training.dataset import (
    extract_offline_features, extract_y_labels
    )


def offline_trainer(do_cv="False"):

    try:
        '''
        features = extract_offline_features(
            in_path="data/xdf_files/francien_calibration.xdf",
            out_path="data/csv_files/calibration2.csv")
        # print("Labeled the features.")

        X,y = extract_y_labels(
            csv_path = "data/csv_files/calibration2.csv",
            in_path="data/xdf_files/francien_calibration.xdf")
        # print("Constructed X and y matrices")
        '''
        features = extract_offline_features(
            in_path="data/xdf_files/matthijs_data_100.xdf",
            out_path="data/csv_files/calibration_matthijs.csv")
        print("Labeled the features.")

        X, y = extract_y_labels(
            csv_path="data/csv_files/calibration_matthijs.csv",
            in_path="data/xdf_files/matthijs_data_100.xdf")

        print("Constructed X and y matrices")

        clf = OnlineLDA()
        clf.train(X, y)

        if X is not None:
            clf.train(X, y)
            print("LDA trained.")
            if do_cv:
                cv_scores = clf.evaluate_lda_cv(X, y)
                print("\n--- 5-Fold Cross-Validation Results ---")
                print(f"Scores per fold: {np.round(cv_scores, 4)}")
                print(f"Mean Accuracy:   {np.mean(cv_scores):.4f}")
                print("Model evaluated.")

    except Exception as error:
        print(f"Could not load or train LDA: {error}")

    return clf


if __name__ == "__main__":
    offline_trainer(do_cv=True)
