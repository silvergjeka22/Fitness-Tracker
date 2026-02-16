from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class ClassificationAlgorithms:

    def forward_selection(self, max_features, X_train, y_train, X_val, y_val):
        """
        Fixed: Now requires separate validation set to avoid data leakage
        """
        selected_features = []
        ordered_scores = []
        print("Starting forward selection...")

        for i in range(max_features):
            best_perf = 0
            best_feature = None
            print(f"Selecting feature {i + 1}/{max_features}...")

            for f in X_train.columns:
                if f not in selected_features:
                    temp_selected_features = selected_features + [f]

                    # FIXED: Evaluate on validation set, not training set
                    _, pred_y_val, _ = self.decision_tree(
                        X_train[temp_selected_features],
                        y_train,
                        X_val[temp_selected_features],
                        y_val,
                        gridsearch=False
                    )

                    # FIXED: Score on validation set
                    perf = accuracy_score(y_val, pred_y_val)

                    if perf > best_perf:
                        best_perf = perf
                        best_feature = f

            if best_feature:
                selected_features.append(best_feature)
                ordered_scores.append(best_perf)

            print(f"After selecting feature {i + 1}/{max_features}, selected features: {selected_features}")

        print("\nForward selection complete.")
        print(f"Final selected features: {selected_features}")
        return selected_features, ordered_scores

    def svm_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True):
        """
        Fixed: Added random_state for reproducibility and improved parameter grid
        """
        if gridsearch:
            svm = GridSearchCV(
                LinearSVC(random_state=42),
                {"max_iter": [1000, 2000], "tol": [1e-3, 1e-4], "C": [1, 5, 10, 20]},
                cv=5, 
                scoring="accuracy",
                n_jobs=-1  # Use all cores for faster training
            )
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter, random_state=42)

        svm.fit(train_X, train_y)

        if gridsearch:
            svm = svm.best_estimator_
            print(f"Best max_iter: {svm.get_params()['max_iter']}")
            print(f"Best tol: {svm.get_params()['tol']}")
            print(f"Best C: {svm.get_params()['C']}")

        pred_train = svm.predict(train_X)
        pred_test = svm.predict(test_X)

        return pred_train, pred_test, svm

    def k_nearest_neighbor(self, train_X, train_y, val_X, val_y, gridsearch=True):
        """
        Fixed: Removed delta logic and use GridSearchCV properly
        """
        if gridsearch:
            param_grid = {
                "n_neighbors": list(range(3, 21, 2)),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
            grid = GridSearchCV(
                KNeighborsClassifier(), 
                param_grid, 
                cv=5, 
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(train_X, train_y)

            # FIXED: Use GridSearchCV's best estimator directly
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV score: {grid.best_score_:.4f}")

        else:
            best_model = KNeighborsClassifier(n_neighbors=5)
            best_model.fit(train_X, train_y)

        return best_model.predict(train_X), best_model.predict(val_X), best_model

    def decision_tree(self, train_X, train_y, val_X, val_y, gridsearch=True):
        """
        Fixed: Removed delta logic and use GridSearchCV properly
        """
        if gridsearch:
            param_grid = {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10, 25],
                "criterion": ["gini", "entropy"]
            }

            grid = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid, 
                cv=5, 
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(train_X, train_y)

            # FIXED: Use GridSearchCV's best estimator directly
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV score: {grid.best_score_:.4f}")

        else:
            best_model = DecisionTreeClassifier(
                min_samples_leaf=10,
                criterion="gini",
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
            best_model.fit(train_X, train_y)

        return best_model.predict(train_X), best_model.predict(val_X), best_model

    def naive_bayes(self, train_X, train_y, test_X):
        """
        No changes needed - Naive Bayes has no hyperparameters to tune
        """
        nb = GaussianNB()
        nb.fit(train_X, train_y)
        return nb.predict(train_X), nb.predict(test_X), nb

    def random_forest(self, train_X, train_y, val_X, val_y, gridsearch=True):
        """
        Fixed: Removed delta logic and use GridSearchCV properly
        """
        if gridsearch:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "min_samples_leaf": [1, 5, 10],
                "max_depth": [5, 10, 20, None],
                "max_features": ["sqrt", "log2"],
                "criterion": ["gini", "entropy"]
            }

            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid, 
                cv=5, 
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(train_X, train_y)

            # FIXED: Use GridSearchCV's best estimator directly
            best_model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV score: {grid.best_score_:.4f}")

        else:
            best_model = RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                criterion="gini",
                max_depth=10,
                max_features="sqrt",
                random_state=42
            )
            best_model.fit(train_X, train_y)

        return best_model.predict(train_X), best_model.predict(val_X), best_model

    def logistic_regression(self, train_X, train_y, test_X, gridsearch=True):
        """
        Fixed: Added more regularization options and random_state
        """
        if gridsearch:
            logreg = GridSearchCV(
                LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
                {
                    'C': [0.01, 0.1, 1, 5, 10],
                    'penalty': ['l1', 'l2']
                },
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
        else:
            logreg = LogisticRegression(solver='liblinear', C=1, max_iter=1000, random_state=42)

        logreg.fit(train_X, train_y)

        if gridsearch:
            logreg = logreg.best_estimator_
            print(f"Best params: {logreg.get_params()}")

        return logreg.predict(train_X), logreg.predict(test_X), logreg