import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# 把所有单独模态下的单独特征整理
# base_name = 'radiomics'
# model_name = 'MRT2'
# file = pd.read_excel('./20230727/radiomics_6_04.xlsx')
# IDs = np.array(file['ID'])
#
# df1 = pd.read_csv('./radiomics/{}_GTVnd_radiomics.csv'.format(model_name, model_name))
# df2 = pd.read_csv('./radiomics/{}_GTVnx_radiomics.csv'.format(model_name, model_name))
# df3 = pd.read_csv('./radiomics/{}_PGTVnd_radiomics.csv'.format(model_name, model_name))
# df4 = pd.read_csv('./radiomics/{}_PGTVnx_radiomics.csv'.format(model_name, model_name))
#
# df_GTVnd = pd.DataFrame(data=None, columns=['Image', *df1.columns[38:]])
# df_GTVnx = pd.DataFrame(data=None, columns=['Image', *df2.columns[38:]])
# df_PGTVnd = pd.DataFrame(data=None, columns=['Image', *df3.columns[38:]])
# df_PGTVnx = pd.DataFrame(data=None, columns=['Image', *df4.columns[38:]])
#
# for index, row in file.iterrows():
#     ID = row['ID']
#     print(ID)
#     df1_ = pd.DataFrame(data=df1[df1['Image'] == ID], columns=['Image', *df1.columns[38:]])
#     df_GTVnd = df_GTVnd.append(df1_, ignore_index=True)
#
#     df2_ = pd.DataFrame(data=df2[df2['Image'] == ID], columns=['Image', *df2.columns[38:]])
#     df_GTVnx = df_GTVnx.append(df2_, ignore_index=True)
#
#     df3_ = pd.DataFrame(data=df3[df3['Image'] == ID], columns=['Image', *df3.columns[38:]])
#     df_PGTVnd = df_PGTVnd.append(df3_, ignore_index=True)
#
#     df4_ = pd.DataFrame(data=df4[df4['Image'] == ID], columns=['Image', *df4.columns[38:]])
#     df_PGTVnx = df_PGTVnx.append(df4_, ignore_index=True)
#
# # df_GTVnd = df_GTVnd.drop('Image', axis=1)
# # df_GTVnx = df_GTVnx.drop('Image', axis=1)
# # df_PGTVn = df_PGTVnd.drop('Image', axis=1)
# df_PGTVnx = df_PGTVnx.drop('Image', axis=1)
#
# # for name in df_GTVnd.columns:
# #     if name == 'Image':
# #         pass
# #     else:
# #         df_GTVnd.rename(columns={name: 'GTVnd_{}'.format(name)}, inplace=True)
# #
# # for name in df_GTVnx.columns:
# #     if name == 'Image':
# #         pass
# #     else:
# #         df_GTVnx.rename(columns={name: 'GTVnx_{}'.format(name)}, inplace=True)
#
# for name in df_PGTVnd.columns:
#     if name == 'Image':
#         pass
#     else:
#         df_PGTVnd.rename(columns={name: 'PGTVnd_{}'.format(name)}, inplace=True)
#
# for name in df_PGTVnx.columns:
#     if name == 'Image':
#         pass
#     else:
#         df_PGTVnx.rename(columns={name: 'PGTVnx_{}'.format(name)}, inplace=True)
#
# # for name in df_PGTVnd.columns:
# #     df_PGTVnd.rename(columns={name: 'PGTVnd_{}'.format(name)}, inplace=True)
# #
# # for name in df_PGTVnx.columns:
# #     df_PGTVnx.rename(columns={name: 'PGTVnx_{}'.format(name)}, inplace=True)
#
# df_all = pd.concat([df_PGTVnd], axis=1)
# # df_all = pd.concat([df_PGTVnx,df_PGTVnd], axis=1)
# print(df_all)
#
# df_all.to_csv('./20230727/{}PGTVnd_{}_use.csv'.format(model_name, base_name), index=False)

# 两两组合——组合的所有特征
base_name = 'radiomics'
model_name1 = 'RTdosePGTV'
model_name2 = 'MRT2PGTV'
model_name = 'MRT2_RTdose_PGTV'
file = pd.read_excel('./20230727/radiomics_7_27.xlsx')
IDs = np.array(file['ID'])
df_c = pd.DataFrame(data=file, columns=['T', 'N'])
print(df_c)
df1 = pd.read_csv('./20230727/{}_{}_use.csv'.format(model_name1, base_name))
df2 = pd.read_csv('./20230727/{}_{}_use.csv'.format(model_name2, base_name))
df2 = df2.drop('Image', axis=1)
for name in df1.columns:
    if name == 'Image':
        pass
    else:
        df1.rename(columns={name: '{}_{}'.format(model_name1, name)}, inplace=True)
for name in df2.columns:
    df2.rename(columns={name: '{}_{}'.format(model_name2, name)}, inplace=True)

df_all = pd.concat([df1, df2, df_c], axis=1)
print(df_all)
df_all.to_csv('./20230727/{}_{}_use.csv'.format(model_name, base_name), index=False)

# MR T2
# df1 = pd.read_csv('./radiomics/MR/MR_1_T2_GTVnx_radiomics.csv')
# df2 = pd.read_csv('./radiomics/MR/MR_2_T2_GTVnx_radiomics.csv')
# df_MR1 = pd.DataFrame(data=None, columns=['Image', *df1.columns[38:]])
# df_MR2 = pd.DataFrame(data=None, columns=['Image', *df2.columns[38:]])
# for index, row in file.iterrows():
#     ID = row['ID']
#     print(ID)
#     df1_ = pd.DataFrame(data=df1[df1['Image'] == ID], columns=['Image', *df1.columns[38:]])
#     df_MR1 = df_MR1.append(df1_, ignore_index=True)
#
#     df2_ = pd.DataFrame(data=df2[df2['Image'] == ID], columns=['Image', *df2.columns[38:]])
#     df_MR2 = df_MR2.append(df2_, ignore_index=True)
#
# df_MR2 = df_MR2.drop('Image', axis=1)
# for name in df_MR1.columns:
#     if name == 'Image':
#         pass
#     else:
#         df_MR1.rename(columns={name: 'MR1_{}'.format(name)}, inplace=True)
#
# for name in df_MR2.columns:
#     df_MR2.rename(columns={name: 'MR2_{}'.format(name)}, inplace=True)
#
# df_all = pd.concat([df_MR1, df_MR2], axis=1)
# print(df_all)
#
# df_all.to_csv('./MRT2GTVnx_{}_use.csv'.format(base_name), index=False)

# ALL
# base_name = 'radiomics'
# model_name1 = 'CTPGTV'
# model_name2 = 'RTdosePGTV'
# model_name3 = 'MRT2PGTV'
# model_name = 'ALL(PGTV)'
# df1 = pd.read_csv('./20230727/{}_{}_use.csv'.format(model_name1, base_name))
# df2 = pd.read_csv('./20230727/{}_{}_use.csv'.format(model_name2, base_name))
# df3 = pd.read_csv('./20230727/{}_{}_use.csv'.format(model_name3, base_name))
# df2 = df2.drop('Image', axis=1)
# df3 = df3.drop('Image', axis=1)
# for name in df1.columns:
#     if name == 'Image':
#         pass
#     else:
#         df1.rename(columns={name: '{}_{}'.format(model_name1, name)}, inplace=True)
# for name in df2.columns:
#     df2.rename(columns={name: '{}_{}'.format(model_name2, name)}, inplace=True)
# for name in df3.columns:
#     df3.rename(columns={name: '{}_{}'.format(model_name3, name)}, inplace=True)
# df_all = pd.concat([df1, df2, df3], axis=1)
# print(df_all)
# df_all.to_csv('./20230727/{}_{}_use.csv'.format(model_name, base_name), index=False)