import os
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import joblib
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import kaplanmeier as km
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

base_name = 'radiomics'
split_name = 'train_val_test_20230604'
model_name = 'MRT2_RTdose_PGTV'
model_name_new = 'MRT2_RTdose_PGTV'
pred_val1s = []
pred_val2s = []
pred_val3s = []
pred_val4s = []
pred_val5s = []
pred_val6s = []
pred_val7s = []

tprs1 = []
aucs1 = []
tprs2 = []
aucs2 = []
tprs3 = []
aucs3 = []
tprs4 = []
aucs4 = []
tprs5 = []
aucs5 = []
tprs6 = []
aucs6 = []
tprs7 = []
aucs7 = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(0, 5):
    fold = i
    print('Fold', i)
    train_val_split = pickle.load(open(r'./20230604/{}.pkl'.format(split_name), 'rb'))
    train_ID = train_val_split['train'][0][fold]
    val_ID = train_val_split['val'][0][fold]
    # test_ID = train_val_split['test'][0]
    clinical_info = pd.read_excel(r'./20230604/radiomics_6_04.xlsx')
    radiomics_original = pd.read_csv(r'./20230604/{}_{}_use.csv'.format(model_name, base_name))
    radiomics_uses_train = pd.DataFrame(data=None, columns=radiomics_original.columns)
    radiomics_uses_val = pd.DataFrame(data=None, columns=radiomics_original.columns)
    # radiomics_uses_test = pd.DataFrame(data=None, columns=radiomics_original.columns)
    train_Y = []
    val_Y = []
    sc_Y = []
    for ID in train_ID:
        radiomics_use_train = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
        radiomics_uses_train = radiomics_uses_train.append(radiomics_use_train)
        train_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
    for ID in val_ID:
        radiomics_use_val = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
        radiomics_uses_val = radiomics_uses_val.append(radiomics_use_val)
        val_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
        sc_Y.append(clinical_info[clinical_info['ID'] == ID]['sc'])
    train_Y = np.array(train_Y)
    val_Y = np.array(val_Y)
    sc_Y = np.array(sc_Y)
    # PreCar_Y = []
    # for ID in test_ID:
    #     radiomics_use_test = pd.DataFrame(radiomics_original[radiomics_original['Image'] == ID])
    #     radiomics_uses_test = radiomics_uses_test.append(radiomics_use_test)
    #     PreCar_Y.append(clinical_info[clinical_info['ID'] == ID]['label'])
    # PreCar_Y = np.array(PreCar_Y)

    for pkl_name in os.listdir(os.path.join('./20230604/trained_models', '{}_{}_{}_{}'.format(base_name, split_name, model_name_new, fold))):
        choosed_features = pickle.load(open(r'./20230604/Lasso_features/{}_{}_{}_{}/{}.pkl'.format(base_name, split_name, model_name_new, fold, pkl_name), 'rb'))
        save_dir = r'./20230604/trained_models/{}_{}_{}_{}/{}'.format(base_name, split_name, model_name_new, fold, pkl_name)
        print(len(choosed_features))

        imp_mean = joblib.load(filename="{}/imp_mean.pkl".format(save_dir))
        scaler = joblib.load(filename="{}/StandardScaler.pkl".format(save_dir))

        radiomics_uses_train = radiomics_uses_train.drop(['Image'], axis=1)
        radiomics_uses_val = radiomics_uses_val.drop(['Image'], axis=1)
        # radiomics_uses_test = radiomics_uses_test.drop(['Image'], axis=1)
        train_X = radiomics_uses_train[choosed_features]
        val_X = radiomics_uses_val[choosed_features]
        # PreCar_X = radiomics_uses_test[choosed_features]

        train_X = imp_mean.transform(train_X)
        val_X = imp_mean.transform(val_X)
        # PreCar_X = imp_mean.transform(PreCar_X)

        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        # PreCar_X = scaler.transform(PreCar_X)

        print(train_X.shape)
        print(val_X.shape)
        # print(PreCar_X.shape)
        PreCar_X = val_X
        PreCar_Y = val_Y
        # ********************** LogisticRegression **********************
        clf = joblib.load(filename="{}/LogisticRegression.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        # pred_val1s.append(pred_val)

        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)

        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs1.append(interp_tpr)
        aucs1.append(AUC_PreCar)
        # aucs1.append(viz.roc_auc)

        print('LogisticRegression: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** SVM **********************
        clf = joblib.load(filename="{}/SVM.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)

        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs2.append(interp_tpr)
        aucs2.append(AUC_PreCar)
        # aucs2.append(viz.roc_auc)

        print('SVM: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** RandomForest **********************
        clf = joblib.load(filename="{}/RandomForest.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs3.append(interp_tpr)
        aucs3.append(AUC_PreCar)
        # aucs3.append(viz.roc_auc)

        print('RandomForest: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** DecisionTree **********************
        clf = joblib.load(filename="{}/DecisionTree.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs4.append(interp_tpr)
        aucs4.append(AUC_PreCar)
        # aucs4.append(viz.roc_auc)

        print('DecisionTree: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** KNN **********************
        clf = joblib.load(filename="{}/KNN.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])

        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs5.append(interp_tpr)
        aucs5.append(AUC_PreCar)
        # aucs5.append(viz.roc_auc)

        print('KNN: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** XgBoost **********************
        clf = joblib.load(filename="{}/XgBoost.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)

        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs6.append(interp_tpr)
        aucs6.append(AUC_PreCar)
        # aucs6.append(viz.roc_auc)

        censoring = np.argmax(pred_val, axis=1)
        y = val_Y + 1
        df = {'time': sc_Y.T[0],
              'Died': censoring,
              'group': y.T[0]}
        df = pd.DataFrame(df)
        time_event = df['time']
        censoring = df['Died']
        y = df['group']

        results = km.fit(time_event, censoring, y)
        km.plot(results)

        print('XgBoost: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

        # ********************** MultinomialNB **********************
        clf = joblib.load(filename="{}/MultinomialNB.pkl".format(save_dir))
        pred_train = clf.predict_proba(train_X)
        AUC_train = roc_auc_score(train_Y, pred_train[:, 1])
        pred_val = clf.predict_proba(val_X)
        AUC_val = roc_auc_score(val_Y, pred_val[:, 1])
        pred_PreCar = clf.predict_proba(PreCar_X)
        AUC_PreCar = roc_auc_score(PreCar_Y, pred_PreCar[:, 1])
        fpr, tpr, _ = roc_curve(PreCar_Y, pred_PreCar[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # viz = RocCurveDisplay.from_estimator(
        #     clf,
        #     PreCar_X,
        #     PreCar_Y,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs7.append(interp_tpr)
        aucs7.append(AUC_PreCar)
        # aucs7.append(viz.roc_auc)

        print('MultinomialNB: {}(Train) {}(Val) {}(PreCar)'.format(AUC_train, AUC_val, AUC_PreCar))

ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
mean_tpr = np.mean(tprs1, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs1)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"LR (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

# std_tpr = np.std(tprs1, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )

mean_tpr = np.mean(tprs2, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs2)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="g",
    label=r"SVM (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
# std_tpr = np.std(tprs2, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )

mean_tpr = np.mean(tprs3, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs3)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="m",
    label=r"RF (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
# std_tpr = np.std(tprs3, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )

mean_tpr = np.mean(tprs4, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs4)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="c",
    label=r"DT (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
# std_tpr = np.std(tprs4, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )

mean_tpr = np.mean(tprs5, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs5)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="yellow",
    label=r"KNN (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
# std_tpr = np.std(tprs5, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
mean_tpr = np.mean(tprs6, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs6)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="r",
    label=r"XgB (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

mean_tpr = np.mean(tprs7, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs7)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="orange",
    label=r"NB (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label)",
)

ax.legend(loc="lower right", shadow=True, fontsize='x-large')
# plt.savefig(r'./20230604/picture_results/Survival_Function_Logrank_1.png', transparent=False)
plt.show()
