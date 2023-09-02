import os
import pickle
import random
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegressionCV, LassoCV, LassoLarsCV, LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

base_name = 'radiomics'
split_name = 'train_test_20230727'
model_name = 'PGTVnx'
model_name_new = 'PGTVnx'
for i in range(0, 5):
    fold = i
    print('Fold', i)
    train_val_split = pickle.load(open(r'./20230727/{}.pkl'.format(split_name), 'rb'))
    train_ID = train_val_split['train'][0][fold]
    val_ID = train_val_split['val'][0][fold]
    # test_ID = train_val_split['test'][0]
    clinical_info = pd.read_excel(r'./20230727/radiomics_7_27.xlsx')
    radiomics_original = pd.read_csv(r'./20230727/{}_{}_use.csv'.format(model_name, base_name))
    radiomics_uses_train = pd.DataFrame(data=None, columns=radiomics_original.columns)
    radiomics_uses_val = pd.DataFrame(data=None, columns=radiomics_original.columns)
    # radiomics_uses_test = pd.DataFrame(data=None, columns=radiomics_original.columns)
    train_Y = []
    val_Y = []
    for ID in train_ID:
        radiomics_use_train = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
        radiomics_uses_train = radiomics_uses_train.append(radiomics_use_train)
        train_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
    for ID in val_ID:
        radiomics_use_val = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
        radiomics_uses_val = radiomics_uses_val.append(radiomics_use_val)
        val_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    # PreCar_Y = []
    # for ID in test_ID:
    #     radiomics_use_test = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
    #     radiomics_uses_test = radiomics_uses_test.append(radiomics_use_test)
    #     PreCar_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
    # PreCar_Y = np.array(PreCar_Y)

    for feature_pkl in os.listdir(os.path.join('./20230727//mRMR_features', '{}_{}_{}_{}'.format(base_name, split_name, model_name_new, fold))):
        choosed_features = pickle.load(open(r'./20230727//mRMR_features/{}_{}_{}_{}/{}'.format(base_name, split_name, model_name_new, fold, feature_pkl), 'rb'))
        print(choosed_features)
        save_dir = r'./20230727/trained_models/{}_{}_{}_{}/{}'.format(base_name, split_name, model_name_new, fold, feature_pkl[:-4])
        os.makedirs(save_dir, exist_ok=True)
        radiomics_uses_train = radiomics_uses_train.drop(['Image'], axis=1)
        radiomics_uses_val = radiomics_uses_val.drop(['Image'], axis=1)
        # radiomics_uses_test = radiomics_uses_test.drop(['Image'], axis=1)
        # train_val_radiomics = train_val_radiomics_original[train_val_radiomics_original.ID.isin(train_ID + val_ID)][choosed_features]
        # train_val_radiomics = train_val_radiomics_original[train_val_radiomics_original.ID.isin(train_ID)][choosed_features]
        train_val_radiomics = radiomics_uses_train[choosed_features]
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(train_val_radiomics)
        joblib.dump(imp_mean, '{}/imp_mean.pkl'.format(save_dir))
        scaler = StandardScaler()
        scaler.fit(train_val_radiomics)
        joblib.dump(scaler, '{}/StandardScaler.pkl'.format(save_dir))

        train_X = radiomics_uses_train[choosed_features]
        val_X = radiomics_uses_val[choosed_features]
        # PreCar_X = radiomics_uses_test[choosed_features]

        train_X = imp_mean.transform(train_X)
        val_X = imp_mean.transform(val_X)
        # PreCar_X = imp_mean.transform(PreCar_X)

        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        # PreCar_X = scaler.transform(PreCar_X)
        PreCar_X = val_X
        PreCar_Y = val_Y

        # ********************** LogisticRegression **********************
        best_AUC = 0
        for sover in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
            for Cs in [5, 10, 100]:
                clf = LogisticRegressionCV(random_state=42, max_iter=1000, class_weight='balanced', cv=10, scoring='roc_auc',
                                           solver=sover, Cs=Cs)
                clf.fit(train_X, train_Y)
                pred_PreCar = clf.predict_proba(PreCar_X)
                AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                if AUC_PreCar > best_AUC:
                    best_AUC = AUC_PreCar
                    print('sover: {}  Cs: {}'.format(sover, Cs))
                    pred_train = clf.predict_proba(train_X)
                    AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                    pred_val = clf.predict_proba(val_X)
                    AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                    print('LogisticRegression: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                    joblib.dump(clf, '{}/LogisticRegression.pkl'.format(save_dir))


        # ********************** SVM **********************
        best_AUC = 0
        for kernel in ['rbf', 'poly', 'sigmoid']:
            for gamma in ['scale', 'auto', 0.05, 1e-1, 1e-2, 1e-3]:
                for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]:
                    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=True, random_state=42)
                    clf.fit(train_X, train_Y)
                    pred_PreCar = clf.predict_proba(PreCar_X)
                    AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                    if AUC_PreCar > best_AUC:
                        best_AUC = AUC_PreCar
                        print('kernel: {}  gamma: {}  C: {}'.format(kernel, gamma, C))
                        pred_train = clf.predict_proba(train_X)
                        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                        pred_val = clf.predict_proba(val_X)
                        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                        print('SVM: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                        joblib.dump(clf, '{}/SVM.pkl'.format(save_dir))
        for kernel in ['linear']:
            for C in [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]:
                clf = svm.SVC(kernel=kernel, C=C, probability=True, random_state=42)
                clf.fit(train_X, train_Y)
                pred_PreCar = clf.predict_proba(PreCar_X)
                AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                if AUC_PreCar > best_AUC:
                    best_AUC = AUC_PreCar
                    print('kernel: {}  C: {}'.format(kernel, C))
                    pred_train = clf.predict_proba(train_X)
                    AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                    pred_val = clf.predict_proba(val_X)
                    AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                    print('SVM: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                    joblib.dump(clf, '{}/SVM.pkl'.format(save_dir))


        # ********************** RandomForest **********************
        best_AUC = 0
        for n_estimators in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for max_depth in [1, 2, 3]:
                for min_samples_split in [2, 3, 4, 5, 6]:
                    for min_samples_leaf in [1, 2, 3, 4, 5]:
                        for max_features in ['auto']:
                            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features,
                            random_state=42, class_weight='balanced', n_jobs=2)
                            clf.fit(train_X, train_Y)
                            pred_PreCar = clf.predict_proba(PreCar_X)
                            AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                            if AUC_PreCar > best_AUC:
                                best_AUC = AUC_PreCar
                                print('n_estimators: {}  max_depth: {}  min_samples_split: {}  min_samples_leaf: {}  '
                                      'max_features: {}'.format(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features))
                                pred_train = clf.predict_proba(train_X)
                                AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                                pred_val = clf.predict_proba(val_X)
                                AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                                print('RandomForest: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                                joblib.dump(clf, '{}/RandomForest.pkl'.format(save_dir))


        # ********************** DecisionTree **********************
        best_AUC = 0
        for criterion in ["gini", "entropy"]:
            for splitter in ["best", "random"]:
                for max_depth in [None, 1, 2, 3, 4, 5]:
                    for min_samples_split in [0.5, 2, 3, 4]:
                        for min_samples_leaf in [1, 2, 3, 4]:
                            for max_features in ["auto", "sqrt", "log2", None]:
                                clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                             max_features=max_features, random_state=42)
                                clf.fit(train_X, train_Y)
                                pred_PreCar = clf.predict_proba(PreCar_X)
                                AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                                if AUC_PreCar > best_AUC:
                                    best_AUC = AUC_PreCar
                                    print('criterion: {}  splitter: {}  max_depth: {}  min_samples_leaf: {}  '
                                          'max_features: {}'.format(criterion, splitter, max_depth,
                                                                    min_samples_leaf, max_features))
                                    pred_train = clf.predict_proba(train_X)
                                    AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                                    pred_val = clf.predict_proba(val_X)
                                    AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                                    print('DecisionTree: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                                    joblib.dump(clf, '{}/DecisionTree.pkl'.format(save_dir))


        # ********************** KNN **********************
        best_AUC = 0
        for n_neighbors in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for weights in ['uniform', 'distance']:
                for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=2)
                    clf.fit(train_X, train_Y)
                    pred_PreCar = clf.predict_proba(PreCar_X)
                    AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                    if AUC_PreCar > best_AUC:
                        best_AUC = AUC_PreCar
                        print('n_neighbors: {}  weights: {}  algorithm: {}'.format(n_neighbors, weights, algorithm))
                        pred_train = clf.predict_proba(train_X)
                        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                        pred_val = clf.predict_proba(val_X)
                        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                        print('KNN: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                        joblib.dump(clf, '{}/KNN.pkl'.format(save_dir))


        # ********************** XgBoost **********************
        best_AUC = 0
        params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'learning_rate': 0.1, 'seed': 42,
                  'eval_metric': 'logloss'}
        for n_estimators in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                for min_child_weight in [1, 2, 3, 4, 5]:
                    for subsample in [0.6, 0.7, 0.8, 0.9, 1]:
                        for colsample_bytree in [0.6, 0.7, 0.8, 0.9, 1]:
                            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                min_child_weight=min_child_weight,
                                                subsample=subsample, colsample_bytree=colsample_bytree, **params)
                            clf.fit(train_X, train_Y)
                            pred_PreCar = clf.predict_proba(PreCar_X)
                            AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
                            if AUC_PreCar > best_AUC:
                                best_AUC = AUC_PreCar
                                print('n_estimators: {}  max_depth: {}  min_child_weight: {}  subsample: {}  '
                                      'colsample_bytree: {}'.format(n_estimators, max_depth, min_child_weight, subsample,
                                                                    colsample_bytree))
                                pred_train = clf.predict_proba(train_X)
                                AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                                pred_val = clf.predict_proba(val_X)
                                AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                                print('XgBoost: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                                joblib.dump(clf, '{}/XgBoost.pkl'.format(save_dir))


        # ********************** MultinomialNB **********************
        best_AUC = 0
        scaler = MinMaxScaler()
        scaler.fit(train_X)
        X_train_ = scaler.transform(train_X)
        val_X_ = scaler.transform(val_X)
        PreCar_X_ = scaler.transform(PreCar_X)
        for alpha in [0.01, 0.1, 0.5, 1, 2, 10]:
            clf = MultinomialNB(alpha=alpha)
            clf.fit(X_train_, train_Y)
            pred_PreCar = clf.predict_proba(PreCar_X_)
            AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
            if AUC_PreCar > best_AUC:
                best_AUC = AUC_PreCar
                print('alpha: {}'.format(alpha))
                pred_train = clf.predict_proba(X_train_)
                AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
                pred_val = clf.predict_proba(val_X_)
                AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
                print('MultinomialNB: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))
                joblib.dump(clf, '{}/MultinomialNB.pkl'.format(save_dir))