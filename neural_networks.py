import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder


# MLPClassifier class
class MLP:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # 1 Hidden Layer MLP Classifier
    def mlpClassifier(self):
        print("\n1 Hidden Layers NN")
        neurons = [50, 100, 200]
        best_cross_validation_score = (0, 0)
        best_train_test_score = (0, 0)
        best_k = 0

        for k in neurons:
            # initialize MLP Classifier
            mlp_classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd')

            # fit classifier on training data
            mlp_classifier.fit(self.X_train, self.y_train)

            # evaluate classifier on test data
            test_preds = mlp_classifier.predict(self.X_test)
            test_acc = accuracy_score(test_preds, self.y_test)
            test_f1 = f1_score(test_preds, self.y_test, average='weighted')
            train_test_score = (test_acc, test_f1)

            # evaluate classifier using cross-validation on training data
            cv_acc_scores = cross_val_score(mlp_classifier, self.X_train, self.y_train, cv=10)
            cv_accuracy = cv_acc_scores.mean()

            cv_f1_scores = cross_val_score(mlp_classifier, self.X_train, self.y_train, cv=10, scoring='f1_weighted')
            cv_f1score = cv_f1_scores.mean()

            cross_validation_score = (cv_accuracy, cv_f1score)

            if cross_validation_score > best_cross_validation_score and train_test_score > best_train_test_score:
                best_cross_validation_score = cross_validation_score
                best_train_test_score = train_test_score
                best_k = k

        print("Best Score")
        print(best_k, "Neurons")
        print("Accuracy:", f'{best_train_test_score[0]:.2f}')
        print("F1 score:",  f'{best_train_test_score[1]:.2f}')
        print("Cross Validation Score")
        print("Mean Accuracy:", f'{best_cross_validation_score[0]:.2f}')
        print("Mean F1 score:", f'{best_cross_validation_score[1]:.2f}')

    # 2 Hidden Layer MLP Classifier
    def mlpClassifierTwo(self):
        print("\n2 Hidden Layers NN")
        n_pairs = [(50, 25), (100, 50), (200, 100)]
        best_cross_validation_score = (0, 0)
        best_train_test_score = (0, 0)
        best_k = 0

        for k in n_pairs:
            # initialize MLP Classifier
            mlp_classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd')

            # fit classifier on training data
            mlp_classifier.fit(self.X_train, self.y_train)

            # evaluate classifier on test data
            test_preds = mlp_classifier.predict(self.X_test)
            test_acc = accuracy_score(test_preds, self.y_test)
            test_f1 = f1_score(test_preds, self.y_test, average='weighted')
            train_test_score = (test_acc, test_f1)

            # evaluate classifier using cross-validation on training data
            cv_acc_scores = cross_val_score(mlp_classifier, self.X_train, self.y_train, cv=10)
            cv_accuracy = cv_acc_scores.mean()

            cv_f1_scores = cross_val_score(mlp_classifier, self.X_train, self.y_train, cv=10, scoring='f1_weighted')
            cv_f1score = cv_f1_scores.mean()

            cross_validation_score = (cv_accuracy, cv_f1score)

            if cross_validation_score > best_cross_validation_score and train_test_score > best_train_test_score:
                best_cross_validation_score = cross_validation_score
                best_train_test_score = train_test_score
                best_k = k

        print("Best Score")
        print(best_k, "Neurons")
        print("Accuracy:", f'{best_train_test_score[0]:.2f}')
        print("F1 score:", f'{best_train_test_score[1]:.2f}')
        print("Cross Validation Score")
        print("Mean Accuracy:", f'{best_cross_validation_score[0]:.2f}')
        print("Mean F1 score:", f'{best_cross_validation_score[1]:.2f}')


""" Mobile Dataset """
mobile_dataset = pd.read_csv("data/train_mobile.csv")
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# split data to train and test sets
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=42)

print("Mobile Data")
mlp_mobile = MLP(x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test)
mlp_mobile.mlpClassifier()
mlp_mobile.mlpClassifierTwo()
