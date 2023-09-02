import pickle
import os
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

base_name = 'radiomics'
model_name = 'RTdosePGTVnd'
split_name = 'train_test_20230727'
for i in range(0, 5):
    fold = i
    print('Fold', i)

    save_dir = os.path.join('./20230727/Lasso_features', '{}_{}_{}_{}'.format(base_name, split_name, model_name, fold))
    os.makedirs(save_dir, exist_ok=True)

    train_val_radiomics = pd.read_csv(r'./20230727/{}_{}_use.csv'.format(model_name, base_name))
    clinical_info = pd.read_excel(r'./20230727/radiomics_7_27.xlsx')
    radiomics_uses = pd.DataFrame(data=None, columns=train_val_radiomics.columns)

    split = pickle.load(open(r'./20230727/{}.pkl'.format(split_name), 'rb'))
    print(split)
    # ID_list = split['train'][fold] + split['val'][fold]
    ID_list = split['train'][0][fold]
    print(len(ID_list))
    Y = []
    for ID in ID_list:
        radiomics_use = pd.DataFrame(train_val_radiomics[train_val_radiomics['Image'] == ID])
        radiomics_uses = radiomics_uses.append(radiomics_use)
        Y.append(clinical_info[clinical_info['ID'] == ID]['label'])

    radiomics_uses = radiomics_uses.drop(['Image'], axis=1)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(radiomics_uses)
    X = StandardScaler().fit_transform(X)
    Y = np.array(Y)

    alphas = np.logspace(-3, 1, 50)

    # model_lasso = LassoCV(alphas=alphas, cv=10, max_iter=100000, normalize=False, random_state=42).fit(X, Y)
    model_lasso = LassoCV(cv=10, random_state=0).fit(X, Y)
    print(model_lasso.alpha_)
    coef = pd.Series(model_lasso.coef_, index=radiomics_uses.columns)
    print(
        "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

    coef_ = model_lasso.coef_
    choosed_features = list(radiomics_uses.columns[np.where(coef_ != 0)])
    print(choosed_features)

    dic = {'Selected_features': radiomics_uses.columns, 'Coefficients': model_lasso.coef_}
    df = pd.DataFrame(dic)
    df1 = df[df['Coefficients'] != 0]
    df1.to_csv(os.path.join('./20230727/Lasso_features', '{}_{}_{}_{}.csv'.format(base_name, split_name, model_name, fold)), index=False)

    coef = pd.Series(model_lasso.coef_, index=radiomics_uses.columns)
    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
    # sns.set(font_scale=1.2)
    # plt.rc('font', family='simsun')
    imp_coef.plot(kind="barh")
    plt.title("Lasso")
    plt.show()

    with open('{}/Lasso_{}.pkl'.format(save_dir, model_lasso.alpha_), 'wb') as f:
        pickle.dump(choosed_features, f, pickle.HIGHEST_PROTOCOL)
