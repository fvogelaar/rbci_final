"""
Offline training script
"""

from src.classifier import OnlineLDA
from src.dataset import extract_offline_features, extract_y_labels  # alternatively, load_csv


def offline_trainer():

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
            in_path="data/xdf_files/matthijs_data_50.xdf",
            out_path="data/csv_files/calibration_matthijs.csv")
        print("Labeled the features.")
        
        X,y = extract_y_labels(
            csv_path = "data/csv_files/calibration_matthijs.csv",
            in_path="data/xdf_files/matthijs_data_50.xdf")
        
        print("Constructed X and y matrices")
        #'''
        clf = OnlineLDA()
        clf.train(X, y)
    
        if X is not None:
            clf.train(X, y)
            # print("LDA trained.")
            cv = clf.evaluate_lda_cv(X, y)
           # print("Model evaluated.")
    except Exception as error:
        print(f"Could not load or train LDA: {error}")
    
    return clf


if __name__ == "__main__":
    offline_trainer()