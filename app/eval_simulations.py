# %%[markdown]
# # Quantitative analysis of grammar-learning simulations

# %%[markdown]
# ## Imports and data loading

# %%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import bambi as bmb
import arviz as az

# %%
crossval_data = pd.read_csv('crossval_data.csv')
crossval_data = crossval_data[crossval_data.runid == 1]
crossval_data.drop('runid', axis=1, inplace=True)
crossval_data.rename(columns = {'acc': 'accuracy'}, inplace=True)
crossval_data

# %%
baseline_data = pd.merge(
    crossval_data[(crossval_data.harmony == "simple") & (crossval_data.rhythm == "none")], 
    pd.read_csv('baseline_data.csv'),
    on = "treeid")
baseline_data.harmony = "none"
baseline_data.accuracy = baseline_data.acc
baseline_data.drop('acc', axis=1, inplace=True)
baseline_data

# %%
data = pd.concat([crossval_data, baseline_data], ignore_index=True)
data

# %%[markdown]
# ## Plot accuracy distributions

# %%
g = sns.catplot(x='rhythm', y="accuracy", hue='harmony', col="mode", row="headedness",
    data=data[data.harmony != "none"], kind='violin', split=True, inner="quartile",
    legend_out=False)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.07, 0.49))

# %%
# g.savefig("accuracy_plot.pdf")

# %%[markdown]
# ## Run Bayesian regression for accuracy prediction

# %%
regression_input = data.copy()
regression_input = regression_input[(regression_input['headedness'] != "left") & 
                                    (regression_input['mode'] == "unsupervised")]
regression_input.drop(["headedness", "mode"], axis=1, inplace=True)
regression_input

# %%
model = bmb.Model('accuracy ~ harmony*rhythm + (1 | treeid) + (1 | foldid)', 
                  regression_input)
model

# %%
results = model.fit(tune=1000, draws=3000, chains=4, target_accept=0.95, cores=1)

# %%
# Key summary and diagnostic info on the model parameters
az.summary(results)

# %%
# Use ArviZ to plot the results
az.plot_trace(data=results, compact=False,
    var_names=["Intercept", "harmony", "rhythm"],
    filter_vars="like"
)
plt.tight_layout()

# %%
baseline = results.posterior["Intercept"].data.flatten()
hsimple  = results.posterior["harmony"][:, :, 0].data.flatten()
htransp  = results.posterior["harmony"][:, :, 1].data.flatten()
rsimple  = results.posterior["rhythm"][:, :, 1].data.flatten()
rregula  = results.posterior["rhythm"][:, :, 0].data.flatten()

hsimple_rsimple = results.posterior["harmony:rhythm"][:, :, 1].data.flatten()
hsimple_rregula = results.posterior["harmony:rhythm"][:, :, 0].data.flatten()
htransp_rsimple = results.posterior["harmony:rhythm"][:, :, 3].data.flatten()
htransp_rregula = results.posterior["harmony:rhythm"][:, :, 2].data.flatten()

# %%
acc_dists = pd.DataFrame({
    'baseline': baseline,
    'hsimple': baseline + hsimple,
    'htranspinv': baseline + htransp,
    # 'rsimple': baseline + rsimple,
    # 'rregula': baseline + rregula,
    'hsimple * rsimple':     baseline + hsimple + rsimple + hsimple_rsimple,
    'htranspinv * rsimple':  baseline + htransp + rsimple + htransp_rsimple,
    'hsimple * rregular':    baseline + hsimple + rregula + hsimple_rregula,
    'htranspinv * rregular': baseline + htransp + rregula + htransp_rregula,
})

sns.set_theme(context="paper", style="whitegrid", font_scale=2.5)
g = sns.displot(data = acc_dists, kind='kde', aspect=2.5, palette="Paired", linewidth=3)
plt.xlim([0, 0.71])

# %%
g.savefig("accdist_plot.pdf")

# %%
def evidence(better, comparison):
    prob = np.mean(acc_dists[better] > acc_dists[comparison])
    odds = prob / (1 - prob)
    return odds

def print_evidence(better, comparison):
    print(better, " > ", comparison, ": ", evidence(better, comparison))

# %%
print_evidence("hsimple", "baseline")
print_evidence("htranspinv", "hsimple")
print_evidence('hsimple * rsimple', "hsimple")
print_evidence('hsimple * rsimple', "htranspinv")
print_evidence('htranspinv * rsimple', 'hsimple * rsimple')
print_evidence('hsimple * rregular', 'hsimple * rsimple')
print_evidence('htranspinv * rregular', 'hsimple * rsimple')

# %%
evidence_df = pd.DataFrame({
    'model': [
        "hsimple", 
        "htranspinv", 
        "hsimple * rsimple", 
        "hsimple * rsimple",
        "htranspinv * rsimple",
        "hsimple * rregular",
        "htranspinv * rregular"
    ],
    'comparison': [
        "baseline",
        "hsimple",
        "hsimple",
        "htranspinv",
        "hsimple * rsimple",
        "hsimple * rsimple",
        'hsimple * rsimple',
    ],
    'evidence for improvement': [
        evidence("hsimple", "baseline"),
        evidence("htranspinv", "hsimple"),
        evidence('hsimple * rsimple', "hsimple"),
        evidence('hsimple * rsimple', "htranspinv"),
        evidence('htranspinv * rsimple', 'hsimple * rsimple'),
        evidence('hsimple * rregular', 'hsimple * rsimple'),
        evidence('htranspinv * rregular', 'hsimple * rsimple'),
    ]
})

def transform_evidence(e):
    if np.isinf(e):
        return '> 1000'
    else:
        return f'{e:.2f}'

evidence_df['evidence for improvement'] = \
    evidence_df['evidence for improvement'].map(transform_evidence)

evidence_df

# %%
print(evidence_df.to_latex(index=False, column_format='rrr'))