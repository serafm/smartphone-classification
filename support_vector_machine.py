from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class SVM:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Linear SVM
    def linearSVM(self):
        # initialize Linear Support Vector Machine
        svm_linear = SVC(kernel='linear')

        # fit classifier on training data
        svm_linear.fit(self.X_train, self.y_train)

        # evaluate classifier on test data
        test_preds = svm_linear.predict(self.X_test)
        test_acc = accuracy_score(test_preds, self.y_test)
        test_f1 = f1_score(test_preds, self.y_test, average='weighted')
        train_test_score = (test_acc, test_f1)

        # evaluate classifier using cross-validation on training data
        cv_acc_scores = cross_val_score(svm_linear, self.X_train, self.y_train, cv=10)
        cv_accuracy = cv_acc_scores.mean()

        cv_f1_scores = cross_val_score(svm_linear, self.X_train, self.y_train, cv=10, scoring='f1_weighted')
        cv_f1score = cv_f1_scores.mean()

        cross_validation_score = (cv_accuracy, cv_f1score)

        print("\nBest Score of Linear SVM")
        print("Accuracy:", f'{train_test_score[0]:.2f}')
        print("F1 score:", f'{train_test_score[1]:.2f}')
        print("Cross Validation Score")
        print("Mean Accuracy:", f'{cross_validation_score[0]:.2f}')
        print("Mean F1 score:", f'{cross_validation_score[1]:.2f}')

    # Gaussian SVM
    def gaussianSVM(self):
        C = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma = [0.001, 0.01, 0.1, 1, 10, 100]
        best_cross_validation_score = (0, 0)
        best_train_test_score = (0, 0)
        best_params = {}

        for c in C:
            for g in gamma:
                # Initialize SVM with RBF kernel and One-vs-All classifier
                svm_gaussian = OneVsRestClassifier(SVC(kernel='rbf', C=c, gamma=g))

                # fit classifier on training data
                svm_gaussian.fit(self.X_train, self.y_train)

                # evaluate classifier on test data
                test_preds = svm_gaussian.predict(self.X_test)
                test_acc = accuracy_score(test_preds, self.y_test)
                test_f1 = f1_score(test_preds, self.y_test, average='weighted')
                train_test_score = (test_acc, test_f1)

                # evaluate classifier using cross-validation on training data
                cv_acc_scores = cross_val_score(svm_gaussian, self.X_train, self.y_train, cv=10)
                cv_accuracy = cv_acc_scores.mean()

                cv_f1_scores = cross_val_score(svm_gaussian, self.X_train, self.y_train, cv=10, scoring='f1_weighted')
                cv_f1score = cv_f1_scores.mean()

                cross_validation_score = (cv_accuracy, cv_f1score)

                print("params: ", c, g)
                print("Accuracy:", f'{train_test_score[0]:.2f}')
                print("F1 score:", f'{train_test_score[1]:.2f}')
                print("Cross Validation Score")
                print("Mean Accuracy:", f'{cross_validation_score[0]:.2f}')
                print("Mean F1 score:", f'{cross_validation_score[1]:.2f}')

                if cross_validation_score > best_cross_validation_score and train_test_score > best_train_test_score:
                    best_cross_validation_score = cross_validation_score
                    best_train_test_score = train_test_score
                    best_params = {'C': c, 'Gamma': g}

        print("\nBest Score of Gaussian SVM")
        print("Best params:", best_params)
        print("Accuracy:", f'{best_train_test_score[0]:.2f}')
        print("F1 score:", f'{best_train_test_score[1]:.2f}')
        print("Cross Validation Score")
        print("Mean Accuracy:", f'{best_cross_validation_score[0]:.2f}')
        print("Mean F1 score:", f'{best_cross_validation_score[1]:.2f}')


""" Mobile Price """
# load data from csv
mobile_dataset = pd.read_csv("data/train_mobile.csv")

y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# split data to train and test sets
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=42)

print("Mobile Dataset")
svm_classifier_mobile = SVM(x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test)
svm_classifier_mobile.linearSVM()
svm_classifier_mobile.gaussianSVM()
