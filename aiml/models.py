"""The algorithms of Module 06, ready to compare.
Every model that needs scaled inputs is wrapped in a Pipeline, so the scaler is fitted on training data only."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
                              AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier, VotingRegressor, VotingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier


def regressors(seed=0):
    m = {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-3)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, weights="distance")),
        "SVR": make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.05)),
        "Decision tree": DecisionTreeRegressor(max_depth=8, random_state=seed),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=seed),
        "Extra trees": ExtraTreesRegressor(n_estimators=300, random_state=seed),
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=seed),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed),
        "Hist. gradient boosting (XGBoost-style)": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed),
        "Neural network (MLP)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), solver="lbfgs", alpha=1e-2, max_iter=5000, random_state=seed)),
    }
    m["Voting (RF + GB + MLP)"] = VotingRegressor([("rf", m["Random forest"]), ("gb", m["Gradient boosting"]), ("nn", m["Neural network (MLP)"])])
    return m


def classifiers(seed=0):
    m = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "SVC": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=seed)),
        "Decision tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, random_state=seed),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=seed),
        "Neural network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=4000, random_state=seed)),
    }
    m["Voting (soft)"] = VotingClassifier([("rf", m["Random forest"]), ("gb", m["Gradient boosting"]), ("nn", m["Neural network (MLP)"])], voting="soft")
    return m

