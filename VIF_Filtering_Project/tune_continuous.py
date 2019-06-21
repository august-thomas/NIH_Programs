## Set up arguments and workspace
import argparse
import sys
import xgboost
import sklearn
import pandas as pd
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randfloat


parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--prune-prefix', type=str, default='N/A', help='Prefix of GenoML run.')
parser.add_argument('--impute-data', type=str, default='median', help='Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].')
parser.add_argument('--max-tune', type=int, default=50, help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
parser.add_argument('--n-cv', type=int, default=5, help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')

args = parser.parse_args()

print("Here is some basic info on the command you are about to run.")
print("Python version info...")
print(sys.version)
print("CLI argument info...")
print("The imputation method you picked is using the column", args.impute_data, "to fill in any remaining NAs. Ideally this is the same as during the training phase.")
print("Working with the dataset and best model corresponding to prefix", args.prune_prefix, "the timestamp from the merge is the prefix in most cases.")
print("Your maximum number of tuning iterations is", args.max_tune, "and if you are concerned about runtime, make this number smaller.")
print("You are running", args.n_cv, "rounds of cross-valdiation, and again... if you are concerned about runtime, make this number smaller.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")

print("")

## Read in the data. Note engine = 'c' to speed up data frame read, consider chunking and concatenating for extremely large datasets later.

### We also cut the phenotype to 0 | 1 here.

prefix = args.prune_prefix

infile = prefix + '.dataForML'
df = pd.read_csv(infile, engine = 'c', sep = '\t')

y_tune = df.PHENO
X_tune = df.drop(columns=['PHENO'])
IDs_tune = X_tune.ID
X_tune = X_tune.drop(columns=['ID'])

best_algo_name_in = prefix + '.best_algorithm.txt'
best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
best_algo = str(best_algo_df.iloc[0,0])

print("From previous analyses in the training phase, we've determined that the best algorithm for this application is", best_algo, " ... so lets tune it up and see what gains we can make!")

print("")

## Quick memory reduce by deleting old data

print("Taking a quick break to take out the garbage and reduce memory consumption!")
import gc
del df
gc.collect()

print("")

## Algorithm imports and list

### Imports for discrete classifiers

from sklearn.metrics import explained_variance_score, mean_squared_error, median_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

### Algorithm list
algorithms = [
	LinearRegression(),
	RandomForestRegressor(),
	AdaBoostRegressor(),
	GradientBoostingRegressor(),
	SGDRegressor(),
	SVR(),
	MLPRegressor(),
	KNeighborsRegressor(),
	BaggingRegressor(),
	XGBRegressor()
	]

if best_algo == 'LinearRegression':
	algo = getattr(sklearn.linear_model, best_algo)()

if  best_algo == 'SGDRegressor':
	algo = getattr(sklearn.linear_model, best_algo)()

if (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
	algo = getattr(sklearn.ensemble, best_algo)()

if best_algo == 'SVR':
	algo = getattr(sklearn.svm, best_algo)(gamma='auto')

if best_algo == 'MLPRegressor':
	algo = getattr(sklearn.neural_network, best_algo)()

if best_algo == 'XGBRegressor':
	algo = getattr(xgboost, best_algo)()

if best_algo == 'KNeighborsRegressor':
	algo = getattr(sklearn.neighbors, best_algo)()

## Begin the tune by setting hyper parameter limits per algorithm

if best_algo == 'LinearRegression':
	hyperparameters = {"penalty": ["l1", "l2"], "C": sp_randint(1, 10)}
	scoring_metric = make_scorer(explained_variance_score)

if  best_algo == 'SGDRegressor':
	hyperparameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "learning_rate": ["constant", "optimal", "invscaling", "adaptive"]}
	scoring_metric = make_scorer(roc_auc_score)

if (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
	hyperparameters = {"n_estimators": sp_randint(1, 1000)}
	scoring_metric = make_scorer(explained_variance_score)

if best_algo == 'SVR':
	hyperparameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": sp_randint(1, 10)}
	scoring_metric = make_scorer(explained_variance_score)
	
if best_algo == 'MLPRegressor':
	hyperparameters = {"alpha": sp_randfloat(0,1), "learning_rate": ['constant', 'invscaling', 'adaptive']}
	scoring_metric = make_scorer(explained_variance_score)

if best_algo == 'XGBRegressor':
	hyperparameters = {"max_depth": sp_randint(1, 100), "learning_rate": sp_randfloat(0,1), "n_estimators": sp_randint(1, 100), "gamma": sp_randfloat(0,1)}
	scoring_metric = make_scorer(explained_variance_score)

if best_algo == 'KNeighborsRegressor':
	hyperparameters = {"leaf_size": sp_randint(1, 100), "n_neighbors": sp_randint(1, 10)}
	scoring_metric = make_scorer(explained_variance_score)

## Now use the randomized search CV for tuning, currently tuning on a max 50 iterations at max cores per job

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

max_iter = args.max_tune
cv_count = args.n_cv

print("Here is a summary of the top 10 iterations of the hyperparameter tune...")

rand_search = RandomizedSearchCV(estimator=algo, param_distributions=hyperparameters, scoring=scoring_metric, n_iter=max_iter, cv=cv_count, n_jobs=-1, random_state=153, verbose=0)
start = time()
rand_search.fit(X_tune, y_tune)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
	" parameter iterations." % ((time() - start), max_iter))

def report(results, n_top=10):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
			results['mean_test_score'][candidate],
			results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

report(rand_search.cv_results_)
rand_search.best_estimator_

## Summarize and save best tuned model then compare it to the baseline

print("Here is the cross-validation summary of your best tuned model hyperparameters...")
cv_tuned = cross_val_score(estimator=rand_search.best_estimator_, X=X_tune, y=y_tune, scoring=scoring_metric, cv=cv_count, n_jobs=-1, verbose=0)
print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
print(cv_tuned)
print("Mean cross-validation score:")
print(cv_tuned.mean())
print("Standard deviation of the cross-validation score:")
print(cv_tuned.std())

print()

print("Here is the cross-validation summary of your baseline/default hyperparamters for the same algorithm on the same data...")
cv_baseline = cross_val_score(estimator=algo, X=X_tune, y=y_tune, scoring=scoring_metric, cv=cv_count, n_jobs=-1, verbose=0)
print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
print(cv_baseline)
print("Mean cross-validation score:")
print(cv_baseline.mean())
print("Stnadard deviation of the cross-validation score:")
print(cv_baseline.std())

print()
print("Just a note, if you have a relatively small variance among the cross-validation iterations, there is a higher chance of your model being more generalizable to similar datasets.")

## Now compare performance

print()
if cv_baseline.mean() > cv_tuned.mean():
	print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
	print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
	print()
	print("Lets shut everythong down, thanks for trying to tune your model with GenoML.")

if cv_baseline.mean() < cv_tuned.mean():
	print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
	print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
	print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")
	
	algo_tuned = rand_search.best_estimator_
	
	### Save it using joblib
	from joblib import dump, load
	algo_tuned_out = prefix + '.tunedModel.joblib'
	dump(algo_tuned, algo_tuned_out)
	
	### Exporting tuned data, which is by nature overfit.
	
	tune_predicted_values = algo_tuned.predict(X_tune)
	tune_predicted_values_df = pd.DataFrame(tune_predicted_values)
	y_tune_df = pd.DataFrame(y_tune)
	IDs_tune_df = pd.DataFrame(IDs_tune)

	tune_out = pd.concat([IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
	tune_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
	tune_out = tune_out.drop(columns=["INDEX"])

	tune_outfile = prefix + '.tunedModel_allSample_Predictions.csv'
	tune_out.to_csv(tune_outfile, index=False)
	
	print("")
	print("Preview of the exported predictions for the tuning samples which is naturally overfit and exported as", tune_outfile, "in the similar format as in the initial training phase of GenoML.")
	print("#"*30)
	print(tune_out.head())
	print("#"*30)
	
	### Export the regression plot
	import seaborn as sns

	plot_out = prefix + '.tunedModel_allSample_ROC.png'

	genoML_colors = ["cyan","purple"]

	sns_plot = sns.regplot(data=tune_out, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"}, line_kws={"color": "purple"})

	sns_plot.figure.savefig(plot_out, dpi=600)

	print()
	print("We are also exporting a regression plot for you here", plot_out, "this is a graphical representation of the model in all samples for the best performing algorithm.")
	
	### Export regression summary

	print()
	print("Here is a quick summary of the regression comparing PHENO_REPORTED ~ PHENO_PREDICTED in the tuned data (all samples so this is overfit) ...")
	print()

	import statsmodels.formula.api as sm
	reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=tune_out)
	fitted = reg_model.fit()
	print(fitted.summary())

	print()
	print("... though always good to see the P for the predictor.")
		
	print()
	print("Lets shut everythong down, thanks for trying to tune your model with GenoML.")
	