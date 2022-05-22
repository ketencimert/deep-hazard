import numpy as np

from auton_survival import datasets, preprocessing, models 
from auton_survival.metrics import survival_diff_metric
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from auton_survival.models.dsm import DeepSurvivalMachines

outcomes, features = datasets.load_dataset("SUPPORT")

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = [key for key in features.keys() if key not in cat_feats]

features = preprocessing.Preprocessor().fit_transform(
    cat_feats=cat_feats, 
    num_feats=num_feats,
    data=features,
    )

x, t, e = features, outcomes.time, outcomes.event

horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e==1], horizons).tolist()

n = len(x)

tr_size = int(n*0.70)
vl_size = int(n*0.10)
te_size = int(n*0.20)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]

param_grid = {'k' : [3, 4, 6],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [ 1e-4, 1e-3],
              'layers' : [ [50], [50, 50], [100], [100, 100] ],
              'discount': [ 1/2, 3/4, 1 ]
             }

params = ParameterGrid(param_grid)

model_dict = {}
for param in params:
    model = DeepSurvivalMachines(k = param['k'],
                                  distribution = param['distribution'],
                                  layers = param['layers'])
    
    # model = models.cph.DeepCoxPH(layers=[100])
    # The fit method is called to train the model
    model, loss = model.fit(
        x_train.values, 
        t_train.values,
        e_train.values, 
        iters = 1000,
        learning_rate = param['learning_rate']
        )
    
    model_dict[model] = np.mean(loss)

model = min(model_dict, key=model_dict.get)

out_risk = model.predict_risk(x_test.values, times)
out_survival = model.predict_survival(x_test.values, times)

cis = []
brs = []

et_train = np.array(
    [(e_train.values[i], t_train.values[i]) for i in range(len(e_train))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array(
    [(e_test.values[i], t_test.values[i]) for i in range(len(e_test))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array(
    [(e_val.values[i], t_val.values[i]) for i in range(len(e_val))],
                 dtype = [('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(
        concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0]
        )
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(
        cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0]
        )
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")