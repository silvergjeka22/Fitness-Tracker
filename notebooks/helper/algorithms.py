from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ClassificationAlgorithms:
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def forward_selection(self, max_features, X_train, y_train, X_val, y_val):
        #greedy forward feature selection using F1 on validation set
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

                    _, pred_y_val, _ = self.decision_tree(
                        X_train[temp_selected_features],
                        y_train,
                        X_val[temp_selected_features],
                        gridsearch=False
                    )

                    perf = f1_score(y_val, pred_y_val, average='weighted')

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

    def svm_without_kernel(self, train_X, train_y, test_X, gridsearch=True):
        #linear SVM with balanced classes
        model = LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
        if gridsearch:
            grid = GridSearchCV(model, {'C': [0.1, 1, 10]}, cv=self.cv, 
                               scoring='f1_weighted', n_jobs=-1)
            grid.fit(train_X, train_y)
            print(f"SVM F1: {grid.best_score_:.4f}")
            model = grid.best_estimator_
        model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(test_X), model

    def k_nearest_neighbor(self, train_X, train_y, val_X, gridsearch=True):
        #KNN with scaling
        pipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])
        if gridsearch:
            grid = GridSearchCV(pipe, {
                'knn__n_neighbors': range(5, 25, 2),
                'knn__weights': ['uniform']
            }, cv=self.cv, scoring='f1_weighted', n_jobs=-1)
            grid.fit(train_X, train_y)
            print(f"KNN F1: {grid.best_score_:.4f}")
            model = grid.best_estimator_
        else:
            model = pipe.set_params(knn__n_neighbors=9, knn__weights='uniform')
            model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

    def decision_tree(self, train_X, train_y, val_X, gridsearch=True):
        #Decision Tree
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        if gridsearch:
            grid = GridSearchCV(model, {
                'max_depth': [3, 5, 8],
                'min_samples_split': [20, 50],
                'min_samples_leaf': [10, 25, 50],
                'criterion': ['gini', 'entropy']
            }, cv=self.cv, scoring='f1_weighted', n_jobs=-1)
            grid.fit(train_X, train_y)
            print(f"DT F1: {grid.best_score_:.4f}")
            model = grid.best_estimator_
        else:
            model.set_params(max_depth=5, min_samples_leaf=20, min_samples_split=30)
            model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

    def naive_bayes(self, train_X, train_y, test_X):
        #Gaussian NB
        model = GaussianNB()
        model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(test_X), model

    def random_forest(self, train_X, train_y, val_X, gridsearch=True):
        # Random Forest with tree constraints to prevent overfitting
        model = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
        if gridsearch:
            grid = GridSearchCV(model, {
                'n_estimators': [100, 200],
                'max_depth': [6, 10, 15],
                'min_samples_leaf': [5, 10, 20],
                'max_features': ['sqrt']
            }, cv=self.cv, scoring='f1_weighted', n_jobs=-1)
            grid.fit(train_X, train_y)
            print(f"RF F1: {grid.best_score_:.4f}")
            model = grid.best_estimator_
        else:
            model.set_params(n_estimators=150, max_depth=12, min_samples_leaf=10)
            model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(val_X), model

    def logistic_regression(self, train_X, train_y, test_X, gridsearch=True):
        #Logistic Regression with L1,L2 regularization
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42, 
                                  class_weight='balanced')
        if gridsearch:
            grid = GridSearchCV(model, {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }, cv=self.cv, scoring='f1_weighted', n_jobs=-1)
            grid.fit(train_X, train_y)
            print(f"LR F1: {grid.best_score_:.4f}")
            model = grid.best_estimator_
        else:
            model.fit(train_X, train_y)
        return model.predict(train_X), model.predict(test_X), model

    def evaluate_model(self, y_true, y_pred, name="Model"):
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"{name}: Acc={acc:.4f} F1w={f1w:.4f}")
