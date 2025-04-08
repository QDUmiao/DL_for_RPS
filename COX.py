import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sksurv.metrics import brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from matplotlib import colors
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy import stats
from sklearn.metrics import roc_auc_score
import scikit_posthocs as sp
import scipy.stats as st

class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):

        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:

        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold :print("There is a significant difference")
        else:        print("There is NO significant difference")




def compute_auc_ci(X, y, va_times, model, n_bootstraps=200, rng_seed=42, alpha=0.95):

    rng = np.random.RandomState(rng_seed)
    bootstrapped_aucs = []

    min_time_idx = np.argmin(y['Survival_in_days'])
    max_time_idx = np.argmax(y['Survival_in_days'])

    for _ in range(n_bootstraps):

        indices = rng.randint(0, len(X), len(X))

        indices = np.unique(np.concatenate(([min_time_idx, max_time_idx], indices)))
        
        X_resampled = X.iloc[indices]
        y_resampled = y[indices]

        if len(np.unique(y_resampled['Status'])) > 1:
            model.fit(X_resampled, y_resampled)
            cph_risk_scores = model.predict(X_resampled)

            cph_auc, _ = cumulative_dynamic_auc(y_resampled, y_resampled, cph_risk_scores, va_times)
            bootstrapped_aucs.append(cph_auc)  
    
    bootstrapped_aucs = np.array(bootstrapped_aucs)
    
    LOW = []
    UP = []
    for i, time in enumerate(va_times):
        time_aucs = bootstrapped_aucs[:, i]

        Q1 = np.percentile(time_aucs, 25)
        Q3 = np.percentile(time_aucs, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_auc = [auc for auc in time_aucs if lower_bound <= auc <= upper_bound]
        
        lower = np.percentile(filtered_auc, (1.0 - alpha) / 2.0 * 100)
        upper = np.percentile(filtered_auc, (1.0 + alpha) / 2.0 * 100)
        
        LOW.append(lower)
        UP.append(upper)

    return LOW,UP



def bootstrap_c_index(X, y, model, n_iterations=1000):
    c_indices = []
    for _ in range(n_iterations):
        X_sample, y_sample = resample(X, y)
        model.fit(X_sample, y_sample)
        c_index = model.score(X_sample, y_sample)
        c_indices.append(c_index)
    

    lower = np.percentile(c_indices, 2.5)
    upper = np.percentile(c_indices, 97.5)
    return round(lower,4), round(upper,4)


def min_max_scaling(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def softmax(z):
    exp_z = np.exp(z - np.max(z)) 
    return exp_z / np.sum(exp_z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def draw_cal(label, PROB, ALLNAME, savepath, savename=""):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for prob, model_name in zip(PROB, ALLNAME):
        fraction_of_positives, mean_predicted_value = calibration_curve(label, prob, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (model_name,),lw=1)
        ax2.hist(prob, range=(0, 1), bins=10, label=model_name, histtype="step", lw=1)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plots (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=1)

    if savename:
        plt.savefig(f'{savepath}\\{savename}.jpg', dpi=200)
    else:
        plt.savefig(f'{savepath}\\CAL.jpg', dpi=200)

    plt.show() 
    

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model
            
def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def draw_DCA(label, PROB, ALLNAME, savepath, savename="",ylim=False):
    thresh_group = np.arange(0, 1, 0.01)
    net_benefit_model = [calculate_net_benefit_model(thresh_group, prob, label) for prob in PROB]
    net_benefit_all = calculate_net_benefit_all(thresh_group, label)
    
    fig, ax = plt.subplots()
    color = ['crimson', 'dodgerblue', 'purple', 'gold', 'green', 'blue', 'darkorange', 'chocolate']
    for i in list(colors.CSS4_COLORS):
        if i not in color:
            color.append(i)
    
    for i, net_benefit in enumerate(net_benefit_model):
        ax.plot(thresh_group, net_benefit, color=color[i % len(color)], label=ALLNAME[i], linestyle='--')
    
    ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all', linewidth=1)
    ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
    
    ax.set_xlim(0, 1)
    if(ylim):
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(max(np.array(net_benefit_model).min() - 0.15,-0.5), np.array(net_benefit_model).max() + 0.15)
    ax.set_xlabel('Threshold Probability', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    ax.set_ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    ax.legend(loc='lower left')
    
    if savename:
        plt.savefig(f'{savepath}\\{savename}.jpg', dpi=300)
    else:
        plt.savefig(f'{savepath}\\DCA.jpg', dpi=300)
    
    plt.show()
    
ALLPATH = []
path = r""
ALLPATH.append(path)
path = path = r""
ALLPATH.append(path)
path = path = r"C"
ALLPATH.append(path)
path = path = r""
ALLPATH.append(path)
path = r""
ALLPATH.append(path)
path = r""
ALLPATH.append(path)
path = r""
ALLPATH.append(path)
path = r""
ALLPATH.append(path)


ALLNAME = ['COMBINED','CLINICAL','RADIOMICS','ITH','DL-F','DL-E2E','DL-F+ITH','DL-F+RADIOMICS']
va_times = np.arange(12, 12*6, 12) 

ALLTRAINDF = []
ALLVALDF = []
ALLTESTDF = []

X_TRAIN = []
Y_TRAIN = []
LABEL_TRAIN = []

X_TEST = []
Y_TEST = []
LABEL_TEST = []

X_VAL = []
Y_VAL = []
LABEL_VAL = []

MODEL = []

PRED_TRAIN = []
PRED_TEST = []
PRED_VAL = []

CINDEX_TRAIN = []
CINDEX_TEST = []
CINDEX_VAL = []

LOWER_TRAIN = []
LOWER_TEST = []
LOWER_VAL = []

UPPER_TRAIN = []
UPPER_TEST = []
UPPER_VAL = []



AUC_TRAIN = []
AUC_TEST = []
AUC_VAL = []

LOW_TRAIN = []
UP_TRAIN = []
LOW_TEST = []
UP_TEST = []
LOW_VAL = []
UP_VAL = []

MEAN_AUC_TRAIN = []
MEAN_AUC_TEST = []
MEAN_AUC_VAL = []

RISK_SCORES_TRAIN = []
RISK_SCORES_TEST = []
RISK_SCORES_VAL = []

IBS_TRAIN = []
IBS_TEST = []
IBS_VAL = [] 


IBSS_TRAIN = []
IBSS_TEST = []
IBSS_VAL = [] 

IBSSTIME_TRAIN = []
IBSSTIME_TEST = []
IBSSTIME_VAL = [] 

for path in ALLPATH:
    
    train_path = ""
    val_path = ""
    test_path = ""

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    
    LABEL_TRAIN.append(train_df['Status'].values)
    LABEL_TEST.append(test_df['Status'].values)
    LABEL_VAL.append(val_df['Status'].values)

    train_df['Status'] = train_df['Status'].replace({0: False, 1: True})
    val_df['Status'] = val_df['Status'].replace({0: False, 1: True})
    test_df['Status'] = test_df['Status'].replace({0: False, 1: True})
    
    max_train_time = train_df['TIME'].max()
    val_df.loc[val_df['TIME'] > max_train_time, 'TIME'] = max_train_time
    test_df.loc[test_df['TIME'] > max_train_time, 'TIME'] = max_train_time
    
    try:
        train_df.drop('coxp', axis=1, inplace=True)
        val_df.drop('coxp', axis=1, inplace=True)
        test_df.drop('coxp', axis=1, inplace=True)
    except:
        pass
    
    y_train = np.array([(row['Status'], row['TIME']) for index, row in train_df.iterrows()],
                       dtype=[('Status', bool), ('Survival_in_days', float)])
    X_train = train_df[['prob']]
    
    y_test = np.array([(row['Status'], row['TIME']) for index, row in test_df.iterrows()],
                       dtype=[('Status', bool), ('Survival_in_days', float)])
    X_test = test_df[['prob']]
    
    y_val = np.array([(row['Status'], row['TIME']) for index, row in val_df.iterrows()],
                       dtype=[('Status', bool), ('Survival_in_days', float)])
    X_val = val_df[['prob']]
    
    
    X_TRAIN.append(X_train)
    Y_TRAIN.append(y_train)
    
    X_TEST.append(X_test)
    Y_TEST.append(y_test)
    
    X_VAL.append(X_val)
    Y_VAL.append(y_val)

    model = CoxPHSurvivalAnalysis()
    model.fit(X_train, y_train)
    MODEL.append(model)
    
    prediction_train = model.predict(X_train)
    
    prediction_test = model.predict(X_test)

    
    prediction_val = model.predict(X_val)
    PRED_TRAIN.append(prediction_train)
    PRED_TEST.append(prediction_test)
    PRED_VAL.append(prediction_val)
    
    
    CINDEX_TRAIN.append(round(model.score(X_train,y_train),4))
    lower, upper = bootstrap_c_index(X_train, y_train, model)
    LOWER_TRAIN.append(lower)
    UPPER_TRAIN.append(upper)
    
    CINDEX_TEST.append(round(model.score(X_test,y_test),4))
    lower, upper = bootstrap_c_index(X_test, y_test, model)
    LOWER_TEST.append(lower)
    UPPER_TEST.append(upper)
    
    
    CINDEX_VAL.append(round(model.score(X_val,y_val),4))
    lower, upper = bootstrap_c_index(X_val, y_val, model)
    LOWER_VAL.append(lower)
    UPPER_VAL.append(upper)
    
    cph_risk_scores = model.predict(X_train)
    RISK_SCORES_TRAIN.append(cph_risk_scores)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_train, cph_risk_scores, va_times)
    AUC_TRAIN.append(cph_auc)
    MEAN_AUC_TRAIN.append(cph_mean_auc)
    lower_train, upper_train = compute_auc_ci(X_train, y_train, va_times, model)

    LOW_TRAIN.append(lower_train)
    UP_TRAIN.append(upper_train)
    
    
    cph_risk_scores = model.predict(X_test)
    RISK_SCORES_TEST.append(cph_risk_scores)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_test, cph_risk_scores, va_times)
    AUC_TEST.append(cph_auc)
    MEAN_AUC_TEST.append(cph_mean_auc)
    lower_test, upper_test = compute_auc_ci(X_test, y_test, va_times, model)
    LOW_TEST.append(lower_test)
    UP_TEST.append(upper_test)

    
    cph_risk_scores = model.predict(X_val)
    RISK_SCORES_VAL.append(cph_risk_scores)
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(y_train, y_val, cph_risk_scores, va_times)
    AUC_VAL.append(cph_auc)
    MEAN_AUC_VAL.append(cph_mean_auc)
    lower_val, upper_val = compute_auc_ci(X_val, y_val, va_times, model)
    LOW_VAL.append(lower_val)
    UP_VAL.append(upper_val)
    
    

    times =  np.linspace(1, 60, len(X_train))
    pred_surv_funcs = model.predict_survival_function(X_train)

    predictions = np.zeros((len(times), len(pred_surv_funcs)))
    for i, surv_func in enumerate(pred_surv_funcs):
        for j, time in enumerate(times):
            predictions[j, i] = surv_func(time)

    ibs = round(integrated_brier_score(y_train, y_train, predictions, times),4)
    IBS_TRAIN.append(ibs)

    SCORE = []
    for i in times:
        preds = [fn(i) for fn in pred_surv_funcs]
        time, score = brier_score(y_train, y_train, preds, i)
        SCORE.append(score)
    IBSS_TRAIN.append(SCORE)
    IBSSTIME_TRAIN.append(times)
    
    times =  np.linspace(1, 60, len(X_test))
    pred_surv_funcs = model.predict_survival_function(X_test)

    predictions = np.zeros((len(times), len(pred_surv_funcs)))
    for i, surv_func in enumerate(pred_surv_funcs):
        for j, time in enumerate(times):
            predictions[j, i] = surv_func(time)

    ibs = round(integrated_brier_score(y_train, y_test, predictions, times),4)
    IBS_TEST.append(ibs)
    
    SCORE = []
    for i in times:
        preds = [fn(i) for fn in pred_surv_funcs]
        time, score = brier_score(y_train, y_test, preds, i)
        SCORE.append(score)
    IBSS_TEST.append(SCORE)
    IBSSTIME_TEST.append(times)
    
    
    times =  np.linspace(3, 60, len(X_val))
    pred_surv_funcs = model.predict_survival_function(X_val)

    predictions = np.zeros((len(times), len(pred_surv_funcs)))
    for i, surv_func in enumerate(pred_surv_funcs):
        for j, time in enumerate(times):
            predictions[j, i] = surv_func(time)

    ibs = round(integrated_brier_score(y_train, y_val, predictions, times),4)
    IBS_VAL.append(ibs)
    
    SCORE = []
    for i in times:
        preds = [fn(i) for fn in pred_surv_funcs]
        time, score = brier_score(y_train, y_val, preds, i)
        SCORE.append(score)
    IBSS_VAL.append(SCORE)
    IBSSTIME_VAL.append(times)
    
print("===============C-INDEX===============")
print("Model        Train                  Test                    Val")
for i in range(len(ALLNAME)):
    print(ALLNAME[i],
          CINDEX_TRAIN[i],"(",LOWER_TRAIN[i],",",UPPER_TRAIN[i],")",
          CINDEX_TEST[i],"(",LOWER_TEST[i],",",UPPER_TEST[i],")",
          CINDEX_VAL[i],"(",LOWER_VAL[i],",",UPPER_VAL[i],")")
print("===============TIME-AUC===============")
for i in range(len(ALLNAME)):
    print(ALLNAME[i],"MEAN-AUC:",MEAN_AUC_TRAIN[i])
    for j in range(len(va_times)):
        print("t:",va_times[j],":",AUC_TRAIN[i][j],"(",LOW_TRAIN[i][j],",",UP_TRAIN[i][j],")")

    
MARKER = ['v','o','^','*','h','x','d','p']
LINEW = [1,1,1,1,1,1,1,1]
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots()

for i, model_name in enumerate(ALLNAME):
    ax.plot(va_times, AUC_TRAIN[i], label=model_name, marker=MARKER[i], linewidth=LINEW[i])

ax.legend()
ax.set_ylim(0, 1)
ax.set_title('AUC by Model Over Time')
ax.set_xlabel('Time (months)')
ax.set_ylabel('AUC')
plt.show()


for i in range(len(ALLNAME)):
    print(ALLNAME[i],"平均AUC:",MEAN_AUC_TEST[i])
    for j in range(len(va_times)):
        print("t:",va_times[j],AUC_TEST[i][j],"(",LOW_TEST[i][j],",",UP_TEST[i][j],")")
        
    
MARKER = ['v','o','^','*','h','x','d','p']
LINEW = [1,1,1,1,1,1,1,1]
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots()

for i, model_name in enumerate(ALLNAME):
    ax.plot(va_times, AUC_TEST[i], label=model_name, marker=MARKER[i], linewidth=LINEW[i])

ax.legend()
ax.set_ylim(0, 1)
ax.set_title('AUC by Model Over Time')
ax.set_xlabel('Time (months)')
ax.set_ylabel('AUC')

plt.show()    
    
    
    

for i in range(len(ALLNAME)):
    print(ALLNAME[i],"MEAN-AUC:",MEAN_AUC_VAL[i])
    for j in range(len(va_times)):
        print("t:",va_times[j],AUC_VAL[i][j],"(",LOW_VAL[i][j],",",UP_VAL[i][j],")")

MARKER = ['v','o','^','*','h','x','d','p']
LINEW = [1,1,1,1,1,1,1,1]
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots()

for i, model_name in enumerate(ALLNAME):
    ax.plot(va_times, AUC_VAL[i], label=model_name, marker=MARKER[i], linewidth=LINEW[i])

ax.legend()
ax.set_ylim(0, 1)
ax.set_title('AUC by Model Over Time')
ax.set_xlabel('Time (months)')
ax.set_ylabel('AUC')

plt.show()    
    


print("===============IBS===============")
    

print("Model        Train       Test        Val")
for i in range(len(ALLNAME)):
    print(ALLNAME[i],
          IBS_TRAIN[i],IBS_TEST[i],IBS_VAL[i])

LINEW = [1,1,1,1,1,1,1,1]
plt.style.use('seaborn-v0_8-bright')

fig, ax = plt.subplots()

for i, model_name in enumerate(ALLNAME):
    ax.plot(IBSSTIME_TRAIN[i], IBSS_TRAIN[i], label=model_name, linestyle='--',linewidth=LINEW[i])

ax.legend()
ax.set_ylim(0, 1)
ax.set_title('Training Set')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Prediction Error')


plt.show()

LINEW = [1,1,1,1,1,1,1,1]
plt.style.use('seaborn-v0_8-bright')

fig, ax = plt.subplots()

for i, model_name in enumerate(ALLNAME):
    ax.plot(IBSSTIME_TEST[i], IBSS_TEST[i], label=model_name, linestyle='--',linewidth=LINEW[i])


ax.legend()
ax.set_ylim(0, 1)
ax.set_title('Test Set')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Prediction Error')

plt.show()


LINEW = [1,1,1,1,1,1,1,1]

plt.style.use('seaborn-v0_8-bright')

fig, ax = plt.subplots()


for i, model_name in enumerate(ALLNAME):
    ax.plot(IBSSTIME_VAL[i], IBSS_VAL[i], label=model_name, linestyle='--',linewidth=LINEW[i])


ax.legend()
ax.set_ylim(0, 1)

ax.set_title('Validation set')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Prediction Error')

plt.show()


PROB_TRAIN = []
PROB_TEST = []
PROB_VAL = []
for i in range(len(ALLNAME)):
    prob = min_max_scaling(PRED_TRAIN[i])
    PROB_TRAIN.append(prob)
    
    prob = min_max_scaling(PRED_TEST[i])
    PROB_TEST.append(prob)
    
    prob = min_max_scaling(PRED_VAL[i])
    PROB_VAL.append(prob)


draw_cal(LABEL_TRAIN[0], PROB_TRAIN, ALLNAME, r'')
draw_cal(LABEL_TEST[0], PROB_TEST, ALLNAME, r'')
draw_cal(LABEL_VAL[0], PROB_VAL, ALLNAME, r'')


draw_DCA(LABEL_TRAIN[0], PROB_TRAIN, ALLNAME,  r'')
draw_DCA(LABEL_TEST[0], PROB_TEST, ALLNAME,  r'')
draw_DCA(LABEL_VAL[0], PROB_VAL, ALLNAME,  r'')

