from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class ClassificationAlgorithms:

    def forward_selection(self, max_features, X_train, y_train):
            selected_features = []  # List to store the selected features
            ordered_scores = []  # List to store the performance scores
            ca = ClassificationAlgorithms()  # Create instance of classification algorithms

            print("Starting forward selection...")

            for i in range(max_features):
                best_perf = 0  # Best performance score for the current iteration
                best_feature = None  # Variable to hold the best feature for the current iteration

                print(f"Selecting feature {i + 1}/{max_features}...")

                for f in X_train.columns:
                    if f not in selected_features:
                        # Temporary list of selected features
                        temp_selected_features = selected_features + [f]
                        
                        # Apply decision tree with the temporary selected features
                        pred_y_train, _, _ = ca.decision_tree(
                            X_train[temp_selected_features], y_train, X_train[temp_selected_features]
                        )
                        
                        # Calculate the performance of the model (accuracy)
                        perf = accuracy_score(y_train, pred_y_train)

                        # Update the best feature if performance improves
                        if perf > best_perf:
                            best_perf = perf
                            best_feature = f  # Update the best feature

                # Append the best feature and its performance to the results
                selected_features.append(best_feature)
                ordered_scores.append(best_perf)

                # Print selected features after each set is finished
                print(f"After selecting feature {i + 1}/{max_features}, selected features: {selected_features}")

            print("\nForward selection complete.")
            print(f"Final selected features: {selected_features}")
            return selected_features, ordered_scores


    def svm_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            svm = GridSearchCV(LinearSVC(), {"max_iter": [1000, 2000, 3000],
                                              "tol": [1e-3, 1e-4],
                                              "C": [5, 10, 15, 20, 25]},
                                              cv=5, scoring="accuracy")
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y)

        # use the best hyperparameters
        if gridsearch:
            svm = svm.best_estimator_
            print(f"Best max_iter: {svm.get_params()['max_iter']}")
            print(f"Best tol: {svm.get_params()['tol']}")
            print(f"Best C: {svm.get_params()['C']}")

        # Predictions
        pred_train = svm.predict(train_X)
        pred_test = svm.predict(test_X)

        return pred_train, pred_test, svm



    def k_nearest_neighbor(self, train_X, train_y, val_X, val_y, gridsearch=True):
        if gridsearch:
            # GridSearchCV
            param_grid = {"n_neighbors": [5, 10, 15, 20, 25, 30, 31, 35, 40]}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", return_train_score=True)
            grid.fit(train_X, train_y)

            # Select the best model
            best_gap = float("inf")
            best_k = None
            best_model = None

            for i in range(len(grid.cv_results_["params"])):
                k = grid.cv_results_["param_n_neighbors"][i]
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(train_X, train_y)

                pred_train = model.predict(train_X)
                pred_val = model.predict(val_X)

                train_acc = accuracy_score(train_y, pred_train)
                val_acc = accuracy_score(val_y, pred_val)
                gap = train_acc - val_acc

                if gap >= 0 and gap < best_gap:
                    best_gap = gap
                    best_k = k
                    best_model = model

            if best_model is not None:
                print(f"Best k: {best_k}")
                pred_train = best_model.predict(train_X)
                pred_val = best_model.predict(val_X)
                return pred_train, pred_val, best_model

        # no gridsearch
        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(train_X, train_y)
        pred_train = model.predict(train_X)
        pred_val = model.predict(val_X)
        return pred_train, pred_val, model



    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion="gini", gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            dtree = GridSearchCV(
                DecisionTreeClassifier(),
                {"min_samples_leaf": [35, 40, 45, 50, 55], "criterion": ["gini", "entropy"]},
                cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        # use the best hyperparameters
        if gridsearch:
            dtree = dtree.best_estimator_

        # Predictions
        pred_train = dtree.predict(train_X)
        pred_test = dtree.predict(test_X)

        return pred_train, pred_test, dtree

    def naive_bayes(self, train_X, train_y, test_X):
        # Create and fit the model
        nb = GaussianNB()
        nb.fit(train_X, train_y)

        # Predictions
        pred_train = nb.predict(train_X)
        pred_test = nb.predict(test_X)
        
        return pred_train, pred_test, nb

    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion="gini", gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            rf = GridSearchCV(
                RandomForestClassifier(),
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 20, 50],
                    "criterion": ["gini", "entropy"]
                },
                cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y)

        # use the best hyperparameters
        if gridsearch:
            rf = rf.best_estimator_
            print(f"Best n_estimators: {rf.get_params()['n_estimators']}")
            print(f"Best min_samples_leaf: {rf.get_params()['min_samples_leaf']}")
            print(f"Best criterion: {rf.get_params()['criterion']}")
            
        # Predictions   
        pred_train = rf.predict(train_X)
        pred_test = rf.predict(test_X)
      
        return pred_train, pred_test, rf
    
    def logistic_regression(self, train_X, train_y, test_X, gridsearch=True):
        if gridsearch:
            logreg = GridSearchCV(
                LogisticRegression(solver='liblinear'),
                {'C': [0.1, 1, 5]},
                cv=5, scoring='accuracy'
            )
        else:
            logreg = LogisticRegression(solver='liblinear', C=5)

        logreg.fit(train_X, train_y)

        if gridsearch:
            logreg = logreg.best_estimator_
            print(f"Best C: {logreg.get_params()['C']}")

        pred_train = logreg.predict(train_X)
        pred_test = logreg.predict(test_X)
        
        return pred_train, pred_test, logreg
