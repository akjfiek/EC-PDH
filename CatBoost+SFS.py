import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
data_df = pd.read_csv('20230930_data.csv')
label_1 = data_df['ddg']
rate = [0.1]
ran=[]
Au=[]
for rat in rate:
    for g in range(1):

        feature_name =['imf_2_dASA_meanValue', 'dssp_b_norm', 'd_ASA_node_1', 'ASA_node_2', 'donor-num', 'dssp_b_shannon',
         'u_ASA_c_56_64_hz', 'imf_3_dASA_meanValue', 'd_ASA_c_24_32_hz', 'd_ASA_node_5', 'ASA_Ed', 'dssp_b_log_energy',
         'imf_2_dASA_energy', 'd-s-ch-avg-DPX', 'DSSPPSI', 'u_ASA_node_5', 'imf_2_dASA_autocorr', 'd_ASA_Ea_2',
         'ASA_c_8_16_hz', 'imf_3_dASA_energy', 'd_ASA_c_1_8_hz', 'dssp_node_7', 'dssp_b_threshold', 'd-Total-Side-ABS',
         'imf_3_dASA_autocorr', 'imf_2_dASA_variance', 'd_ASA_Ea_3', 'ASA_b_log_energy', 'dssp_node_8', 'd_ASA_Ea_1',
         'd-All-atoms-ABS', 'd_ASA_coef_ave', 'imf_3_dASA_variance', 'd_ASA_node_3', 'u_ASA_b_log_energy',
         'ASA_coef_std', 'dssp_Ea_1', 'd_ASA_c_48_56_hz', 'imf_1_dASA_variance', 'd_ASA_c_16_24_hz',
         'imf_1_dASA_energy', 'imf_1_DSSP_meanValue', 'Total-Side-REL', 'imf_1_dASA_autocorr', 'd_ASA_coef_std',
         'u_ASA_node_7', 'd_ASA_sure', 'd-All-polar-ABS', 'dssp_coef_ave', 'd_ASA_node_4', 'ASA_shannon',
         'dssp_c_32_40_hz', 'u_ASA_c_24_32_hz', 'ASA_b_threshold', 'u_ASA_Ea_2', 'd_ASA_node_6', 'dssp_c_40_48_hz',
         'ASA_c_56_64_hz', 'd-s-ch-avg-CX', 'd-Total-Side-REL', 'dssp_Ed', 'u_ASA_b_threshold', 'u_ASA_sure',
         'dssp_sum_hz', 'u_ASA_c_48_56_hz', 'd-All-atoms-REL', 'DSSPTCO', 'ASA_node_5', 'imf_3_ASA_energy',
         'u_ASA_b_norm', 'dssp_node_2', 'dssp_node_1', 'd-average-CX', 'd-Non-polar-ABS', 'u_ASA_node_3',
         'dssp_c_8_16_hz', 'imf_2_uASA_meanValue', 'u_ASA_c_32_40_hz', 'imf_3_ASA_autocorr', 'dssp_b_sure',
         'd-average-DPX', 'd_ASA_node_2', 'imf_2_ASA_variance', 'imf_3_uASA_meanValue', 'u_ASA_Ea_3', 'd_ASA_Ed',
         'u_ASA_sum_hz', 'd-All-polar-REL', 'd-Non-polar-REL', 'ASA_b_shannon', 'imf_2_uASA_variance', 'u_ASA_b_sure',
         'imf_3_ASA_meanValue', 'u_ASA_c_16_24_hz', 'ASA_sure', 'd_ASA_log_energy', 'dssp_threshold', 'u_ASA_threshold',
         'u-All-polar-ABS', 'd_ASA_shannon', 'imf_1_uASA_variance', 'dssp_Ea_2', 'u-s-ch-avg-CX', 'imf_1_uASA_energy',
         'imf_1_uASA_autocorr', 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'imf_2_uASA_energy', 'u-Total-Side-ABS',
         'imf_2_uASA_autocorr', 'u_ASA_node_4', 'u-s-ch-avg-DPX', 'dssp_c_1_8_hz', 'u-All-atoms-ABS', 'All-polar-REL',
         'ASA_b_norm', 'u_ASA_node_8', 'u-average-CX', 'dssp_node_5', 'DSSPACC', 'u_ASA_coef_std', 'u-average-DPX',
         'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'imf_1_ASA_energy', 'd_ASA_sum_hz', 'imf_1_ASA_autocorr',
         'd_ASA_b_sure', 'All-atoms-REL', 'ASA_norm', 'dssp_node_6', 'd_ASA_b_norm', 'dssp_c_48_56_hz', 'ASA_Ea_2',
         'u_ASA_node_6', 'imf_1_ASA_variance', 'ASA_c_16_24_hz', 'Total-Side-ABS', 'imf_3_uASA_energy',
         'imf_3_uASA_autocorr', 's-ch-avg-DPX', 'DSSPPHI', 'd_ASA_b_threshold', 'ASA_log_energy', 'd_ASA_threshold',
         'ASA_node_8', 'imf_1_DSSP_energy', 'ASA_sum_hz', 'd_ASA_norm', 'imf_1_DSSP_autocorr', 's-ch-avg-CX',
         'imf_3_uASA_variance', 'ASA_b_sure', 'Non-polar-REL', 'dssp_sure', 'u-Non-polar-ABS', 'dssp_c_16_24_hz',
         'DSSPKAPPA', 'dssp_c_24_32_hz', 'ASA_Ea_3', 'u_ASA_norm', 'Non-polar-ABS', 'ASA_node_4', 'u_ASA_node_2',
         'u_ASA_Ed', 'u-All-atoms-REL', 'imf_3_ASA_variance', 'ASA_c_24_32_hz', 'All-atoms-ABS', 'u_ASA_c_40_48_hz',
         'u_ASA_c_8_16_hz', 'dssp_coef_std', 'imf_1_uASA_meanValue', 'd_ASA_c_32_40_hz', 'u_ASA_shannon',
         'ASA_threshold', 'u-All-polar-REL', 'ASA_node_3', 'd_ASA_node_8', 'u-Non-polar-REL', 'd_ASA_node_7',
         'u_ASA_c_1_8_hz', 'imf_2_ASA_energy', 'imf_2_ASA_autocorr', 'imf_1_dASA_meanValue', 'u-Total-Side-REL',
         'dssp_Ea_3', 'average-CX', 'All-polar-ABS', 'd_ASA_b_shannon', 'imf_2_DSSP_variance', 'ASA_c_40_48_hz',
         'imf_1_DSSP_variance', 'ASA_c_1_8_hz', 'imf_2_DSSP_energy', 'imf_2_DSSP_autocorr', 'imf_1_ASA_meanValue',
         'DSSPALPHA', 'imf_2_DSSP_meanValue', 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1',
         'u_ASA_Ea_1', 'dssp_shannon', 'imf_2_ASA_meanValue', 'ASA_c_32_40_hz', 'dssp_log_energy', 'ASA_node_1',
         'ASA_Ea_1', 'ASA_coef_ave', 'average-DPX', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7',
         'dssp_norm', 'u_ASA_log_energy' , 'ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']

        i = 0
        A = []
        a = 0
        A.append(a)
        a = 1
        E = []
        B = []

        while (i < 20):
            print(i)
            b = i
            sen1 = []
            spe1 = []
            pre1 = []
            F11 = []
            MCC1 = []
            ACC1 = []
            AUC1 = []
            if feature_name[i - 1] == 'ddg':
                break
            while (i > 0):
                if (A[a - 1] < max(A)):
                    dele = feature_name[i - 1]
                    feature_name.remove(dele)
                    print(dele)
                    feature_name.append(dele)
                    i = i
                    break
                else:
                    E.append(feature_name[i - 1])
                    i = i + 1

                    break
            a = a + 1

            if i == 0:
                i = i + 1

            print(i)
            train_feature = data_df.drop(feature_name[i:len(feature_name)], axis=1)
            # i=25
            # i = i + 1
            if i == 20:
                i = i + 88
            print(train_feature)
            print("max_AUC:", max(A))
            from imblearn.combine import SMOTETomek

            smo = SMOTETomek(random_state=None)
            X, y = smo.fit_resample(train_feature, label_1)
            print(X)
            print(y)

            X = X.reset_index(drop=True).values
            y = y.reset_index(drop=True).values


            for s in range(10):
                acc = []
                Spe = []
                Sen = []
                pre = []
                rec = []
                F1 = []
                MCC = []
                AUC = []
                n = 10

                stratified_folder = StratifiedKFold(n_splits=10, shuffle=False)
                for train_index, test_index in stratified_folder.split(X, y):

                    X_train = X[train_index]
                    Y_train = y[train_index]
                    x_test = X[test_index]
                    y_test = y[test_index]


                    def train_model(X_train, Y_train, x_test, y_test, Sen, Spe, acc, pre, rec, F1, MCC, AUC, rate):
                        catboost_model = CatBoostClassifier(
                            iterations=150,
                            learning_rate=0.03,
                            depth=6,
                            eval_metric='AUC',
                            random_seed=42,
                            bagging_temperature=0,
                            od_type='Iter',
                            od_wait=100
                        )
                        catboost_model.fit(X_train, Y_train, eval_set=(x_test, y_test), use_best_model=True,verbose=True)

                        test_pred = catboost_model.predict_proba(x_test)[:, 1]
                        train_pred = catboost_model.predict_proba(X_train)[:, 1]

                        test_roc_auc_score = roc_auc_score(y_test, test_pred)
                        train_roc_auc_score = roc_auc_score(Y_train, train_pred)
                        print(test_pred)

                        for q in range(len(test_pred)):
                            if test_pred[q] < 0.5:
                                test_pred[q] = 0
                            else:
                                test_pred[q] = 1

                        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
                        if accuracy_score(y_test, test_pred) > 0 or test_roc_auc_score > 0:
                            spe = tn / (tn + fp)
                            sen = tp / (tp + fn)
                            Spe.append(spe)
                            Sen.append(sen)
                            AUC.append(test_roc_auc_score)

                            acc.append(accuracy_score(y_test, test_pred))
                            pre.append(metrics.precision_score(y_test, test_pred))
                            F1.append(metrics.f1_score(y_test, test_pred))
                            MCC.append(matthews_corrcoef(y_test, test_pred))



                            evals_result = {}
                            evals_result['test_roc_auc_score'] = test_roc_auc_score
                            evals_result['train_roc_auc_score'] = train_roc_auc_score
                            return catboost_model, evals_result


                    model, evals_result = train_model(X_train, Y_train, x_test, y_test, Sen, Spe, acc, pre, rec, F1, MCC, AUC, rat)
                sen1.append(np.sum(Sen) / n)
                spe1.append(np.sum(Spe) / n)
                pre1.append(np.sum(pre) / n)
                F11.append(np.sum(F1) / n)
                MCC1.append(np.sum(MCC) / n)
                ACC1.append(np.sum(acc) / n)
                AUC1.append(np.sum(AUC) / n)

            print("sen1", np.sum(sen1) / n, "spe1:", np.sum(Spe) / n, "pre1:", np.sum(pre) / n, "F11:", np.sum(F1) / n,
                  "MCC1:", np.sum(MCC) / n, "ACC1:", np.sum(acc) / n, "AUC1:", np.sum(AUC) / n)

            y = np.sum(AUC1) / n
            A.append(y)
            print("max_AUC:", max(A))
            c = b

            print(A)

        print(A)
        print(B)
        del (A[-1])
        print(max(A))
        ran.append(g)
        Au.append(max(A))
        print(a)
        print(b)
        print(max(Au))
        print(ran[Au.index(max(Au))])
        print(Au)
        print(ran)
        print(train_feature)
        print(E)
        import os
        with open('test_2.txt', 'a') as file0:
            print("Au:"+str(max(A))+"ran"+str(g), file=file0)
            print(str(E)+"\n", file=file0)

print(max(Au))
print(ran[Au.index(max(Au))])
print(ran)
print(Au)
print(train_feature)
print(X)
print(E)
print(Au)

