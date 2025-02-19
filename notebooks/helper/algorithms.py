from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class ClassificationAlgorithms:

    def forward_selection(self, max_features, X_train, y_train):
    
        selected_features = []
        ordered_scores = []
        ca = ClassificationAlgorithms()

        for i in range(max_features):
            best_perf = 0 # best performance
            best_feature = None

            for f in X_train.columns:
                if f not in selected_features:
                    # temporary selected features
                    temp_selected_features = selected_features + [f]
                    # apply decision tree
                    pred_y_train, _ = ca.decision_tree(
                        X_train[temp_selected_features], y_train, X_train[temp_selected_features]
                    )
                    # calculate the performance
                    perf = accuracy_score(y_train, pred_y_train)
                    
                    if perf > best_perf:
                        best_perf = perf
                        best_feature = f # update the best feature

            # append the results
            selected_features.append(best_feature)
            ordered_scores.append(best_perf)

        return selected_features, ordered_scores

    
    def svm_with_kernel(self, train_X, train_y, test_X, kernel="rbf", C=1, gamma=1e-3, gridsearch=True):
        
        # using GridSearchCV tuning
        if gridsearch:
            svm = GridSearchCV(SVC(probability=True), {"kernel": ["rbf", "poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]}, cv=5, scoring="accuracy")
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        
        # Fit the model
        svm.fit(train_X, train_y)

        # use the best estimator
        if gridsearch:
            svm = svm.best_estimator_

        # Predictions
        pred_train = svm.predict(train_X)
        pred_test = svm.predict(test_X)

        return pred_train, pred_test



    def svm_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            svm = GridSearchCV(LinearSVC(), {"max_iter": [1000, 2000], "tol": [1e-3, 1e-4], "C": [1, 10, 100]}, cv=5, scoring="accuracy")
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y)

        # use the best estimator
        if gridsearch:
            svm = svm.best_estimator_

        # Predictions
        pred_train = svm.predict(train_X)
        pred_test = svm.predict(test_X)

        return pred_train, pred_test

    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            knn = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 2, 5, 10]}, cv=5, scoring="accuracy")
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y)

        # use the best estimator
        if gridsearch:
            knn = knn.best_estimator_

        # Predictions
        pred_train = knn.predict(train_X)
        pred_test = knn.predict(test_X)

        return pred_train, pred_test


    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion="gini", gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            dtree = GridSearchCV(
                DecisionTreeClassifier(),
                {"min_samples_leaf": [2, 10, 50, 100, 200], "criterion": ["gini", "entropy"]},
                cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y)

        # use the best estimator
        if gridsearch:
            dtree = dtree.best_estimator_

        # Predictions
        pred_train = dtree.predict(train_X)
        pred_test = dtree.predict(test_X)

        return pred_train, pred_test

    def naive_bayes(self, train_X, train_y, test_X):
        # Create and fit the model
        nb = GaussianNB()
        nb.fit(train_X, train_y)

        # Predictions
        pred_train = nb.predict(train_X)
        pred_test = nb.predict(test_X)
        
        return pred_train, pred_test

    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion="gini", gridsearch=True):
        # GridSearchCV tuning
        if gridsearch:
            rf = GridSearchCV(
                RandomForestClassifier(),
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"]
                },
                cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        rf.fit(train_X, train_y)

        # use the best estimator
        if gridsearch:
            rf = rf.best_estimator_

        # Predictions   
        pred_train = rf.predict(train_X)
        pred_test = rf.predict(test_X)
      

        return pred_train, pred_test
