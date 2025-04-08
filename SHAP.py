import pandas as pd  
import statsmodels.api as sm  
import numpy as np  
import shap  
import matplotlib.pyplot as plt  

def clean_filename(name):  
    """Clean filename, replace illegal characters"""  
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:  
        name = name.replace(char, '_')  
    return name  

train_path = ""  

train_df = pd.read_csv(train_path)  

features = train_df.columns.drop(['id', 'label'])  

X_all = sm.add_constant(train_df[features])  
y = train_df['label']  

model_all = sm.Logit(y, X_all).fit(disp=0, maxiter=100)  
print(model_all.summary())  

params = model_all.params  
conf = model_all.conf_int()  
conf['OR'] = params  
conf.columns = ['2.5%', '97.5%', 'OR']  
conf_exp = np.exp(conf)  
print("OR and 95%CI for all features:")  
print(conf_exp)  

X = train_df[features] 

coef = model_all.params.values[1:]
explainer = shap.LinearExplainer(  
    (coef, model_all.params.values[0]),
    X  
)  

shap_values = explainer.shap_values(X)  

plt.figure(figsize=(12, 10))  
shap.summary_plot(shap_values, X, plot_type="bar", show=False)  
plt.title("Feature Importance (All Features)", fontsize=16)  
plt.tight_layout()  
plt.savefig("shap_importance_all_features.pdf", dpi=300, bbox_inches='tight')  
plt.close()  

plt.figure(figsize=(12, 8))  
shap_importance = np.abs(shap_values).mean(0)  
feature_names = X.columns  

shap_mean = shap_values.mean(0)  
colors = []  

for value in shap_mean[np.argsort(shap_importance)]:  
    if value > 0:  
        colors.append('#ff0d57') 
    else:  
        colors.append('#1E88E5') 

idx = np.argsort(shap_importance)  
plt.barh(range(len(idx)), shap_importance[idx], color=colors)  

plt.yticks(range(len(idx)), [feature_names[i] for i in idx])  
plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12)  
plt.title('Feature Importance Ranking (SHAP Original Colors)', fontsize=16)  

from matplotlib.patches import Patch  
legend_elements = [  
    Patch(facecolor='#ff0d57', label='Positive Impact (Increases Prediction)'),  
    Patch(facecolor='#1E88E5', label='Negative Impact (Decreases Prediction)')  
]  
plt.legend(handles=legend_elements, loc='lower right')  

plt.tight_layout()  
plt.savefig("shap_bar_importance_original_colors.pdf", dpi=300, bbox_inches='tight')  
plt.close()  

plt.figure(figsize=(14, 12))  
ax = shap.summary_plot(shap_values, X, show=False)  
plt.xlim(-3, 3)  # Limit x-axis range, ignore extreme values  
plt.title("Feature SHAP Value Distribution", fontsize=16)  
plt.tight_layout()  
plt.savefig("shap_summary_all_features_limited_range.pdf", dpi=300, bbox_inches='tight')  
plt.close()  


plt.figure(figsize=(14, 10))  
shap.summary_plot(  
    shap_values,   
    X,  
    plot_type="dot",  
    show=False,  
    color_bar=True,  
)  
plt.title("Feature SHAP Value Distribution", fontsize=18, pad=20)  
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)  
plt.tight_layout()  
plt.savefig("shap_summary_enhanced.pdf", dpi=300, bbox_inches='tight')  
plt.close()  

feature_importance = np.abs(shap_values).mean(0)  
feature_names = X.columns  
indices = np.argsort(feature_importance)  
sorted_features = [feature_names[i] for i in indices[-10:]] 



import os  


output_dir = "shap_force_plots_pdf"
os.makedirs(output_dir, exist_ok=True) 

for sample_idx in range(len(X)):   
    plt.figure(figsize=(16, 4)) 
    shap.force_plot(  
        explainer.expected_value,   
        shap_values[sample_idx, :],   
        X.iloc[sample_idx, :], 
        matplotlib=True,
        show=False
    )  

    output_path = os.path.join(output_dir, f"sample_{sample_idx}.pdf")  
    plt.savefig(output_path, dpi=300, bbox_inches="tight") 
    plt.close()  
print(f"All force plots have been saved as PDF files to folder: {output_dir}")  

print("SHAP analysis completed, all charts saved.")  

fig = plt.figure(figsize=(20, 12), facecolor='#f9f9f9')  
plt.style.use('seaborn-v0_8-whitegrid')  

ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)  

shap_importance = np.abs(shap_values).mean(0)  
idx = np.argsort(shap_importance)[-15:] 
feature_labels = [feature_names[i] for i in idx]  

cmap = plt.colormaps['viridis'] 
norm = plt.Normalize(shap_importance[idx].min(), shap_importance[idx].max())  
colors = [cmap(norm(value)) for value in shap_importance[idx]]  


bars = ax1.barh(range(len(idx)), shap_importance[idx], color=colors, height=0.7)  
ax1.set_yticks(range(len(idx)))  
ax1.set_yticklabels(feature_labels, fontsize=12)  
ax1.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold')  
ax1.set_title('Feature Importance Ranking', fontsize=18, fontweight='bold', pad=15)  
ax1.grid(axis='x', linestyle='--', alpha=0.3)  
ax1.spines['top'].set_visible(False)  
ax1.spines['right'].set_visible(False)  

for i, bar in enumerate(bars):  
    value = shap_importance[idx[i]]  
    ax1.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)  

plt.subplot2grid((1, 5), (0, 2), colspan=3)  

sorted_idx = np.argsort(shap_importance)
shap_values_sorted = shap_values[:, sorted_idx[-15:]]  
X_sorted = X.iloc[:, sorted_idx[-15:]]  
shap.summary_plot(  
    shap_values_sorted,  
    X_sorted,  
    plot_type="dot",  
    show=False,  
    max_display=15,  
    cmap=plt.colormaps['viridis'], 
    alpha=0.8  
)  


plt.gca().set_yticklabels([])  
plt.gca().set_ylabel('')  
  
fig.suptitle("Feature Importance Analysis", fontsize=22, fontweight='bold', y=0.98)  

fig.text(0.5, 0.01,   
         "Left: Average impact magnitude of each feature | Right: Distribution of each feature's impact across samples",   
         ha='center', fontsize=12, style='italic')  
 
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.subplots_adjust(wspace=0.2)  

plt.savefig("shap_combined_visualization_enhanced.pdf", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())  
plt.close()  
