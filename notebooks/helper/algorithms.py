from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class ClassificationAlgorithms:

    def forward_selection(self, max_features, X_train, y_train):
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

                    pred_y_train, _, _ = self.decision_tree(
                        X_train[temp_selected_features],
                        y_train,
                        X_train[temp_selected_features],
                        y_train,
                        gridsearch=False
                    )

                    perf = accuracy_score(y_train, pred_y_train)

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
        if gridsearch:
            svm = GridSearchCV(
                LinearSVC(),
                {"max_iter": [1000, 2000, 3000], "tol": [1e-3, 1e-4], "C": [5, 10, 15, 20, 25]},
                cv=5, scoring="accuracy"
            )
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        svm.fit(train_X, train_y)

        if gridsearch:
            svm = svm.best_estimator_
            print(f"Best max_iter: {svm.get_params()['max_iter']}")
            print(f"Best tol: {svm.get_params()['tol']}")
            print(f"Best C: {svm.get_params()['C']}")

        pred_train = svm.predict(train_X)
        pred_test = svm.predict(test_X)

        return pred_train, pred_test, svm

    def k_nearest_neighbor(self, train_X, train_y, val_X, val_y, gridsearch=True, delta=0.01):
        if gridsearch:
            param_grid = {"n_neighbors": list(range(5, 41, 5))}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", return_train_score=True)
            grid.fit(train_X, train_y)

            best_model = None
            best_val_acc = 0

            for i in range(len(grid.cv_results_["params"])):
                k = grid.cv_results_["param_n_neighbors"][i]
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(train_X, train_y)

                train_acc = accuracy_score(train_y, model.predict(train_X))
                val_acc = accuracy_score(val_y, model.predict(val_X))
                gap = train_acc - val_acc

                if gap <= delta and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model

            if best_model:
                print(f"Best k: {best_model.n_neighbors}")
                return (
                    best_model.predict(train_X),
                    best_model.predict(val_X),
                    best_model
                )

        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

    def decision_tree(self, train_X, train_y, val_X, val_y, gridsearch=True, delta=0.01):
        if gridsearch:
            param_grid = {
                "min_samples_leaf": [25, 35, 50],
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10]
            }

            grid = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid, cv=5, scoring="accuracy", return_train_score=True
            )
            grid.fit(train_X, train_y)

            best_model = None
            best_val_acc = 0

            for params in grid.cv_results_["params"]:
                model = DecisionTreeClassifier(**params, random_state=42)
                model.fit(train_X, train_y)

                train_acc = accuracy_score(train_y, model.predict(train_X))
                val_acc = accuracy_score(val_y, model.predict(val_X))
                gap = train_acc - val_acc

                if gap <= delta and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model

            if best_model:
                return (
                    best_model.predict(train_X),
                    best_model.predict(val_X),
                    best_model
                )

        model = DecisionTreeClassifier(
            min_samples_leaf=50, criterion="gini", max_depth=5,
            min_samples_split=5, random_state=42
        )
        model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

    def naive_bayes(self, train_X, train_y, test_X):
        nb = GaussianNB()
        nb.fit(train_X, train_y)
        return nb.predict(train_X), nb.predict(test_X), nb

    def random_forest(self, train_X, train_y, val_X, val_y, gridsearch=True, delta=0.01):
        if gridsearch:
            param_grid = {
                "n_estimators": [50, 100],
                "min_samples_leaf": [5, 10, 25],
                "max_depth": [5, 10, 20],
                "max_features": ["sqrt"],
                "criterion": ["gini", "entropy"]
            }

            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid, cv=5, scoring="accuracy", return_train_score=True, n_jobs=-1
            )
            grid.fit(train_X, train_y)

            best_model = None
            best_val_acc = 0

            for params in grid.cv_results_["params"]:
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(train_X, train_y)

                train_acc = accuracy_score(train_y, model.predict(train_X))
                val_acc = accuracy_score(val_y, model.predict(val_X))
                gap = train_acc - val_acc

                if gap <= delta and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model

            if best_model:
                print(f"Best n_estimators: {best_model.n_estimators}")
                print(f"Best min_samples_leaf: {best_model.min_samples_leaf}")
                print(f"Best max_depth: {best_model.max_depth}")
                print(f"Best max_features: {best_model.max_features}")
                return (
                    best_model.predict(train_X),
                    best_model.predict(val_X),
                    best_model
                )

        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, criterion="gini",
                                       max_depth=10, max_features="sqrt", random_state=42)
        model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

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

        return logreg.predict(train_X), logreg.predict(test_X), logreg