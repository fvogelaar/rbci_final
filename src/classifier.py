"""
Online-safe classifier wrapper
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

class OnlineLDA:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            return None
       
        label = self.model.predict([x])[0]
        label_to_id = {
            "LEFT": 0,
            "RIGHT": 1,}

        return label_to_id[label]

    # Addition: predict probabilities. Currently unused
    def pred_probability(self, x):
        """
        output shape: [P(LEFT), P(RIGHT)]
        it's called for each feature window.
        """
        probabilities = self.model.predict_proba([x])[0]

        # Match each probability to its actual class label
        # we access the model's 'classes' attribute that sklearn makes
        by_label = dict(zip(self.model.classes_, probabilities))

        return [
            float(by_label["LEFT"]),
            float(by_label["RIGHT"]),]

    # Addition: crossvalidation
    def evaluate_lda_cv(self, X, y, n_splits=5, random_state=42):
        model = self.model
        # cross validation setup for class balance
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
            )
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        print(f"Cross-Validation Scores ({n_splits}-Fold): {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f} (± {scores.std() * 2:.4f})")

        return scores

