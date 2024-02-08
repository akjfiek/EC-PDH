names = ['EC-PDH',
         'WTL-PDH',
         'PrPDH',
         'sxPDH',
         'inpPDH',
         ]

sampling_methods = [clf_lr,
                    clf_rf,
                    clf_xgb,
                    clf_adb,
                    clf_gbdt,
                    clf_lgbm
                   ]

colors = ['crimson',
          'orange',
          'gold',
          'mediumseagreen',
          'steelblue',
          'mediumpurple'
         ]

#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, colors, X_train, y_train, save = True)
train_roc_graph.savefig('ROC_Train_all.png')
