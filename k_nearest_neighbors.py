import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def k_neighbors_classifier(X_train, X_test, y_train, y_test):
    # k-neighbors
    neighbors = [1, 3, 5, 10]
    best_cross_validation_score = (0, 0)
    best_train_test_score = (0, 0)
    best_k = 0

    for k in neighbors:
        # initialize k-nearest neighbors classifier
        knn_classifier = KNeighborsClassifier(k, weights="distance")

        # fit classifier on training data
        knn_classifier.fit(X_train, y_train)

        # evaluate classifier on test data
        test_preds = knn_classifier.predict(X_test)
        test_acc = accuracy_score(test_preds, y_test)
        test_f1 = f1_score(test_preds, y_test, average='weighted')
        train_test_score = (test_acc, test_f1)

        # evaluate classifier using cross-validation on training data
        cv_acc_scores = cross_val_score(knn_classifier, X_train, y_train, cv=10)
        cv_accuracy = cv_acc_scores.mean()

        cv_f1_scores = cross_val_score(knn_classifier, X_train, y_train, cv=10, scoring='f1_weighted')
        cv_f1score = cv_f1_scores.mean()

        cross_validation_score = (cv_accuracy, cv_f1score)

        if cross_validation_score > best_cross_validation_score and train_test_score > best_train_test_score:
            best_cross_validation_score = cross_validation_score
            best_train_test_score = train_test_score
            best_k = k

    print("Best Score")
    print(best_k, "Nearest Neighbors")
    print("Accuracy:", f'{best_train_test_score[0]:.2f}')
    print("F1 score:",  f'{best_train_test_score[1]:.2f}')
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
k_neighbors_classifier(x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test)
