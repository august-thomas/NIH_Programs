## Set up arguments and workspace
import argparse          #Consider using the argparse attributes like usage and epilog to assist user at the command line.
import sys
import xgboost
import sklearn
import pandas as pd
import numpy as np
import time

parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--prune-prefix', type=str, default='N/A', help='Prefix of GenoML run.')
parser.add_argument('--impute-data', type=str, default='median', help='Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].')
parser.add_argument('--rank-features', type=str, default='skip', help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow with huge numbers of features [default: skip].')

args = parser.parse_args()

print("")

print("Here is some basic info on the command you are about to run.")
print("Python version info...")
print(sys.version)
print("CLI argument info...")
print("Are you ranking features, even though it is pretty slow? Right now, genoML runs general recursive feature ranking. You chose to", args.rank_features, "this part.")
print("The imputation method you picked is using the column", args.impute_data, "to fill in any remaining NAs.")
print("Working with dataset", args.prune_prefix, "the timestamp from the merge is the prefix in most cases.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")

print("")

## Read in the data. Note engine = 'c' to speed up data frame read, consider chunking and concatenating for extremely large datasets later.

### We also cut the phenotype to 0 | 1 here.

prefix = args.prune_prefix
infile = prefix + '.dataForML'
df = pd.read_csv(infile, engine = 'c', sep = '\t')

print("")

print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
print("#"*30)
print(df.describe())
print("#"*30)

print("")

## Now impute the missing data but first convert to matrix

impute_type = args.impute_data

if impute_type == 'mean': 
	df = df.fillna(df.mean())

if impute_type == 'median':
	df = df.fillna(df.median())

print("")

print("You have just imputed your data, covering up NAs with the column", impute_type, "so that analyses don't crash due to missing data.")
print("Now your data might look a little better (showing the first few lines of the left-most and right-most columns)...")
print("#"*30)
print(df.describe())
print("#"*30)

print("")

## Split the dat into train and test and bank sample IDs

from sklearn.model_selection import train_test_split
y = df.PHENO
X = df.drop(columns=['PHENO'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70:30
IDs_train = X_train.ID
IDs_test = X_test.ID
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

## Quick memory reduce by deleting old data

print("")
print("Taking a quick break to take out the garbage and reduce memory consumption!")
import gc
del df
gc.collect()
print("")

### Imports
from sklearn.metrics import explained_variance_score, mean_squared_error, median_absolute_error, r2_score
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
	
## Now lets compete the algorithms!

print("")
print("Now let's compete these algorithms!")
print("We'll update you as each algorithm runs, then summarize at the end.")
print("Here we test each algorithm under default settings using the same training and test datasets derived from a 70% training and 30% testing split of your data.")
print("For each algorithm, we will output the following metrics...")
print("Algorithm name, hoping that's pretty self-explanatory. Plenty of resources on these common ML algorithms at https://scikit-learn.org and https://xgboost.readthedocs.io/.")
print("explained_variance_score, this is the variance explained by the model per algorithm (scale from 0 to 1 with 1 being completely explained).")
print("mean_squared_error, this is the mean squared error from regression loss.")
print("median_absolute_error, median absolute error from regression loss.")
print("r2_score, standard r2 metric from linear regression (coefficient of determination), remember, this can be negative if your model is really bad.")
print("We also log the runtimes per algorithm.")

print("")

print("Algorithm summaries incoming...")

print("")

log_cols=["Algorithm", "Explained_variance_score", "Mean_squared_error", "Median_absolute_error", "R2_score", "Runtime_Seconds"]
log_table = pd.DataFrame(columns=log_cols)

for algo in algorithms:
	
	start_time = time.time()
	
	algo.fit(X_train, y_train)
	name = algo.__class__.__name__

	print("#"*30)
	print(name)

	test_predictions = algo.predict(X_test)
	test_predictions = test_predictions
	evs = explained_variance_score(y_test, test_predictions)
	print("Explained variance score: {:.4}".format(evs))
	
	test_predictions = algo.predict(X_test)
	test_predictions = test_predictions
	mse = mean_squared_error(y_test, test_predictions)
	print("Mean squared error: {:.4}".format(mse))
	
	test_predictions = algo.predict(X_test)
	test_predictions = test_predictions
	mae = median_absolute_error(y_test, test_predictions)
	print("Median absolut error: {:.4}".format(mae))
	
	test_predictions = algo.predict(X_test)
	test_predictions = test_predictions
	r2s = r2_score(y_test, test_predictions)
	print("R^2 score: {:.4}".format(r2s))
	
	end_time = time.time()
	elapsed_time = (end_time - start_time)
	print("Runtime in seconds: {:.4}".format(elapsed_time)) 

	log_entry = pd.DataFrame([[name, evs, mse, mae, r2s, elapsed_time]], columns=log_cols)
	log_table = log_table.append(log_entry)

print("#"*30)

print("")

log_outfile = prefix + '.training_withheldSamples_performanceMetrics.csv'

print("This table below is also logged as", log_outfile, "and is in your current working directory...")
print("#"*30)
print(log_table)
print("#"*30)

log_table.to_csv(log_outfile, index=False)

## Save the model with the best AUC. First save the algorithm name for future use, then the model itself.

### Saving best performing algorithm name

best_performing_summary = log_table[log_table.Explained_variance_score == log_table.Explained_variance_score.max()]
best_algo = best_performing_summary.at[0,'Algorithm']

print("")

print("Based on your withheld samples, the algorithm with the highest explained variance score is the", best_algo, "... lets save that model for you.")

best_algo_name_out = prefix + '.best_algorithm.txt'
file = open(best_algo_name_out,'w')
file.write(best_algo)
file.close() 

### Rebuilding best performing model and exporting.

### Remeber to pull attributes from text file.

if best_algo == 'LinearRegression':
	algo = getattr(sklearn.linear_model, best_algo)()

if  best_algo == 'SGDRegressor':
	algo = getattr(sklearn.linear_model, best_algo)()

if (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
	algo = getattr(sklearn.ensemble, best_algo)()

if best_algo == 'SVR':
	algo = getattr(sklearn.svm, best_algo)()

if best_algo == 'MLPRegressor':
	algo = getattr(sklearn.neural_network, best_algo)()

if best_algo == 'XGBRegressor':
	algo = getattr(xgboost, best_algo)()

if best_algo == 'KNeighborsRegressor':
	algo = getattr(sklearn.neighbors, best_algo)()

algo.fit(X_train, y_train)
name = algo.__class__.__name__

print("...remember, there are occasionally slight fluxuations in model performance on the same withheld samples...")

print("#"*30)

print(name)

test_predictions = algo.predict(X_test)
test_predictions = test_predictions
evs = explained_variance_score(y_test, test_predictions)
print("Explained variance score: {:.4}".format(evs))

test_predictions = algo.predict(X_test)
test_predictions = test_predictions
mse = mean_squared_error(y_test, test_predictions)
print("Mean squared error: {:.4}".format(mse))

test_predictions = algo.predict(X_test)
test_predictions = test_predictions
mae = median_absolute_error(y_test, test_predictions)
print("Median absolut error: {:.4}".format(mae))

test_predictions = algo.predict(X_test)
test_predictions = test_predictions
r2s = r2_score(y_test, test_predictions)
print("R^2 score: {:.4}".format(r2s))

### Save it using joblib
from joblib import dump, load
algo_out = prefix + '.trainedModel.joblib'
dump(algo, algo_out)

print("#"*30)

print("... this model has been saved as", algo_out, "for later use and can be found in your working directory.")

# Export predicitons separately for training and withheld samples from the test set.

### Exporting withheld test data

test_predicted_values = algo.predict(X_test)
test_predicted_values_df = pd.DataFrame(test_predicted_values)
y_test_df = pd.DataFrame(y_test)
IDs_test_df = pd.DataFrame(IDs_test)

test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
test_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
test_out = test_out.drop(columns=["INDEX"])

test_outfile = prefix + '.trainedModel_withheldSample_Predictions.csv'
test_out.to_csv(test_outfile, index=False)

print("")
print("Preview of the exported predictions for the withheld test data that has been exported as", test_outfile, "these are pretty straight forward.")
print("They generally include the sample ID, the previously reported phenotype and the predicted phenotype from that algorithm,")
print("#"*30)
print(test_out.head())
print("#"*30)


### Exporting training data, which is by nature overfit.

train_predicted_values = algo.predict(X_train)
train_predicted_values_df = pd.DataFrame(train_predicted_values)
y_train_df = pd.DataFrame(y_train)
IDs_train_df = pd.DataFrame(IDs_train)

train_out = pd.concat([IDs_train_df.reset_index(), y_train_df.reset_index(drop=True), train_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
train_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
train_out = train_out.drop(columns=["INDEX"])

train_outfile = prefix + '.trainedModel_trainingSample_Predictions.csv'
train_out.to_csv(train_outfile, index=False)

print("")
print("Preview of the exported predictions for the traiing samples which is naturally overfit and exported as", train_outfile, "in the similar format as in the withheld test dataset that was just exported.")
print("#"*30)
print(train_out.head())
print("#"*30)


### Export the regression plot in witheld samples.

import seaborn as sns

plot_out = prefix + '.trainedModel_withheldSample_ROC.png'

genoML_colors = ["cyan","purple"]

sns_plot = sns.regplot(data=test_out, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"}, line_kws={"color": "purple"})

plot_out = prefix + '.trainedModel_withheldSample_regression.png'
sns_plot.figure.savefig(plot_out, dpi=600)

print()
print("We are also exporting a regression plot for you here", plot_out, "this is a graphical representation of the difference between the reported and predicted phenotypes in the withheld test data for the best performing algorithm.")

### Export regression summary

print()
print("Here is a quick summary of the regression comparing PHENO_REPORTED ~ PHENO_PREDICTED in the withheld test data...")
print()

import statsmodels.formula.api as sm
reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=test_out)
fitted = reg_model.fit()
print(fitted.summary())

print()
print("... always good to see the P for the predictor.")

## Export feature ranks.

print("")

feature_trigger = args.rank_features

if (feature_trigger == 'run'):

	if (best_algo == 'SVR') or (best_algo == 'KNeighborsRegressor'):

		print("Even if you selected to run feature ranking, you can't generate feature ranks using SVR or KNeighborsRegressor ... it just isn't possible.")

	else:
		print("Processing feature ranks, this can take a while. But you will get a relative rank for every feature in the model.")

		from sklearn.feature_selection import RFE

		top_ten_percent = (len(X_train)//10)
		# core_count = args.n_cores
		names = list(X_train.columns)
		rfe = RFE(estimator=algo)
		rfe.fit(X_train, y_train)
		rfe_out = zip(rfe.ranking_, names)
		rfe_df = pd.DataFrame(rfe_out, columns = ["RANK","FEATURE"])
		table_outfile = prefix + '.trainedModel_trainingSample_featureImportance.csv'
		rfe_df.to_csv(table_outfile, index=False)

		print("Feature ranks exported as", table_outfile, "if you want to be very picky and make a more parsimonious model with a minimal feature set, extract all features ranked 1 and rebuild your dataset. This analysis also gives you a concept of the relative importance of your features in the model.")

print()
print("Thanks for training a model with GenoML!")
print()

