# The following program is more-or-less in a working state, but needs to be altered such that all the potential models
# are trained and tested separately on different nodes in the BioWolf HPC - that seems to be main goal of the project.
# Altercations will be written by August Thomas with aid from Dr. Michael Nells

# Firstly, set up arguments and workspace:

import argparse
import sys
import xgboost
import sklearn
import pandas as pd
import numpy as np
import time

# Create an ArgumentParser object with three arguments of string type, and describe them.
parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--prune-prefix', type=str, default='N/A', help='Prefix of GenoML run.')
parser.add_argument('--impute-data', type=str, default='median', help='Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].')
parser.add_argument('--rank-features', type=str, default='skip', help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow with huge numbers of features [default: skip].')

# Assign the object to variable args, and call the .parse_args() method. This parses the arguments and returns a NameSpace object with them.
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
print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case. This may differ from your phenotype file input file, if it is coded 1 or 2, but don't worry, genoML will figure it out.")

print("")

## Read in the data. Note engine = 'c' to speed up data frame read, consider chunking and concatenating for extremely large datasets later.

### We also cut the phenotype to 0 | 1 here.

# Code Block 2:
# The 1st argument on the CMD line is passed to .prune_prefix, which then is assigned to 'prefix' which is made an ArgumentParser object.
# Next we add a string to the end of that prefix in a copy of the object.
# Pandas method .read_csv() now reads the object into a DataFrame object with the C engine, with regex delimiter '\t'
# If statement is used to swap (1, 2) phenotype notation with (0, 1) if it isn't already.

prefix = args.prune_prefix
infile = prefix + '.dataForML'
df = pd.read_csv(infile, engine = 'c', sep = '\t')
if df.PHENO.max() == 2:
	df['PHENO'] = df['PHENO'] - 1

print("")

# Code Block 3
# Generate descriptive statistics for the data in the DataFrame object. I'm not sure what the ("#"*30) does. Print out
# just a line of hashtags to make the printout neat?

print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
print("#"*30)
print(df.describe())
print("#"*30)

print("")

# Code Block 4
# Impute missing data after converting the DataFrame object .csv into a matrix; first create another copy of the
# NameSpace object 'args' while replacing NaN values, which are inferred by existing non-NaN data. The new NameSpace
# object called 'impute_type.' Detail about potential imputing methods are provided to the user when the system is run.
# Either the mean or median of all values in the DataFrame object 'df' are used to replace NaN values.

impute_type = args.impute_data

if impute_type == 'mean': 
	df = df.fillna(df.mean())

if impute_type == 'median':
	df = df.fillna(df.median())

print("")

# Code Block 5
# Confirm user understands the imputing type and why this was necessary, then print the imputed 'df' DataFrame object.

print("You have just imputed your data, covering up NAs with the column", impute_type, "so that analyses don't crash due to missing data.")
print("Now your data might look a little better (showing the first few lines of the left-most and right-most columns)...")
print("#"*30)
print(df.describe())
print("#"*30)

print("")

# Code Block 6: objective is to split the DataFrame into train and test cases, and then to bank sample IDs for later.
# First import train_test_split method from model_selection class, part of the sklearn library.
# Create DataFrame objects 'y' and 'X'; 'y' corresponding to column 'PHENO' and 'X' to the whole DataFrame sans 'PHENO.'
# I think 'X' is the training case and 'y' is the testing case.
# Instantiate random train and test DataFrame subsets by splitting 'X' and 'y' DataFrames; 30% of the data is used in
# test subset (70% in the train subset) using RNG seed of 42 (the meaning of life).
# Remove the IDs of newly created 'X' training/testing cases after assigning the ID columns to new IDs_train and
# IDs_test DataFrame objects, stored for later use.

from sklearn.model_selection import train_test_split
y = df.PHENO
X = df.drop(columns=['PHENO'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70:30
IDs_train = X_train.ID
IDs_test = X_test.ID
X_train = X_train.drop(columns=['ID'])
X_test = X_test.drop(columns=['ID'])

# Code Block 7: Import and use the gc class to reduce overhead by deleting old and un-split df DataFrame.

print("")
print("Taking a quick break to take out the garbage and reduce memory consumption!")
import gc
del df
gc.collect()
print("")

# Code Block 8: Import classes from various sklearn sub-libraries, and the XGBClassifier class from xgboost library.
# Then, instantiate list 'algorithms' containing each of the classes (probably in order to easily invoke them).

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

algorithms = [
	LogisticRegression(),
	RandomForestClassifier(),
	AdaBoostClassifier(),
	GradientBoostingClassifier(),
	SGDClassifier(loss='modified_huber'),
	SVC(probability=True),
	ComplementNB(),
	MLPClassifier(),
	KNeighborsClassifier(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis(),
	BaggingClassifier(),
	XGBClassifier()
	]

# Code Block 9: Inform user that sklearn class algorithms will now compete against each other.
# Provide an overview of the statistics each algorithm will produce to aid in interpretation.

print("")
print("Now let's compete these algorithms!")
print("We'll update you as each algorithm runs, then summarize at the end.")
print("Here we test each algorithm under default settings using the same training and test datasets derived from a 70% training and 30% testing split of your data.")
print("For each algorithm, we will output the following metrics...")
print("Algorithm name, hoping that's pretty self-explanatory. Plenty of resources on these common ML algorithms at https://scikit-learn.org and https://xgboost.readthedocs.io/.")
print("AUC_percent, this is the area under the curve from reciever operating characteristic analyses. This is the most common metric of classifier performance in biomedical literature, we express this as a percent. We calculate AUC based on the predicted probability of being a case.")
print("Accuracy_percent, this is the simple accuracy of the classifer, how many predictions were correct from best classification cutoff (python default).")
print("Balanced_Accuracy_Percent, consider this as the accuracy resampled to a 1:1 mix of cases and controls. Imbalanced datasets can give funny results for simple accuracy.")
print("Log_Loss, this is essentially the inverse of the likelihood function for a correct prediction, you want to minimize this.")
print("Sensitivity, proportion of cases correcctly identified.")
print("Specificity, proportion of controls correctly identified.")
print("PPV, this is the positive predictive value, the probability that subjects with a positive result actually have the disease.")
print("NPV, this is the negative predictive value, the probability that subjects with a negative result don't have the disease.")
print("We also log the runtimes per algorithm.")

print("")

print("Algorithm summaries incoming...")

print("")

# Code Block 10:
# Instantiate a list of headers (one for each aforementioned summary statistic) and assign the list to columns of empty
# DataFrame object log_table.

log_cols=["Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
log_table = pd.DataFrame(columns=log_cols)

# Code Block 11: For each of the sklearn classes in list algorithms ...
# 1): Make note of run-time.
# 2): Use the .fit() method (which I assume is inherited by every imported class from sklearn, not the sub-libraries).
# 3): Now fit training data we prepared; remember that 'y' was the 'PHENO' DataFrame and 'X' was the sample DataFrame.
# 4): We save the name attribute of whatever sklearn class used in a given algorithm to 'name' and print it.
# 5): A series of temp variables are used in combination with inherited sklearn methods to derive several
# test statistics described to the user including AUC Percent, Accuracy Percent, Balanced Accuracy Percent, & Log Loss.
# 6): Confusion matrix is derived. Used to find probabilities of false-positives, false-negatives, and true predictions.
# Those values are translated into other statistics: specificity, sensitivity, and true positive/negative proportions.
# 7): Finally the header and resulting statistics are appended to log_entry DataFrame, which is printed.
# 8): The resulting DataFrame is exported to a new .csv file with the resulting algorithm name as prefix.
# Notes): Most statistics are carried to four decimal places. Some labels include a *100 to get a x/100 percentage.

for algo in algorithms:
	
	start_time = time.time()
	
	algo.fit(X_train, y_train)
	name = algo.__class__.__name__

	print("#"*30)
	print(name)

	test_predictions = algo.predict_proba(X_test)
	test_predictions = test_predictions[:, 1]
	rocauc = roc_auc_score(y_test, test_predictions)
	print("AUC: {:.4%}".format(rocauc))

	test_predictions = algo.predict(X_test)
	acc = accuracy_score(y_test, test_predictions)
	print("Accuracy: {:.4%}".format(acc))

	test_predictions = algo.predict(X_test)
	balacc = balanced_accuracy_score(y_test, test_predictions)
	print("Balanced Accuracy: {:.4%}".format(balacc))
	
	CM = confusion_matrix(y_test, test_predictions)
	TN = CM[0][0]
	FN = CM[1][0]
	TP = CM[1][1]
	FP = CM[0][1]
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	PPV = TP/(TP+FP)
	NPV = TN/(TN+FN)
	

	test_predictions = algo.predict_proba(X_test)
	ll = log_loss(y_test, test_predictions)
	print("Log Loss: {:.4}".format(ll))
	
	end_time = time.time()
	elapsed_time = (end_time - start_time)
	print("Runtime in seconds: {:.4}".format(elapsed_time)) 

	log_entry = pd.DataFrame([[name, rocauc*100, acc*100, balacc*100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=log_cols)
	log_table = log_table.append(log_entry)

print("#"*30)

print("")

log_outfile = prefix + '.training_withheldSamples_performanceMetrics.csv'

print("This table below is also logged as", log_outfile, "and is in your current working directory...")
print("#"*30)
print(log_table)
print("#"*30)

log_table.to_csv(log_outfile, index=False)

# Code Block 12): Save the model with the best AUC. First save the algorithm name for future use, then the model itself.
# 1): First compare the current algorithm's AUC Percent to the best one found so far; replace if necessary.
# Then assign the index found at [0,'Algorithm'] (I.E the name of the algorithm) to string 'best_algo'.
# 2): Save the corresponding model with distinguishable name, by creating a file and writing the name of the
# applicable algorithm (set as 'best_algo') to it.

best_performing_summary = log_table[log_table.AUC_Percent == log_table.AUC_Percent.max()]
best_algo = best_performing_summary.at[0,'Algorithm']

print("")

print("Based on your withheld samples, the algorithm with the best AUC is the", best_algo, "... lets save that model for you.")

best_algo_name_out = prefix + '.best_algorithm.txt'
file = open(best_algo_name_out,'w')
file.write(best_algo)
file.close() 

# Code Block 13: Rebuilding the best performing model and exporting.
# 1): Check the name of the best algorithm.
# 2): Depending on which algorithm was selected, replace list of sklearn class objects 'algo' with the attributes
# of the top-scoring algorithm via getattr() method. Now 'algo' should contain the values of the named attributes
# of the sklearn class object, instead of a list of different sklearn class objects.
# 3): Reconstruct the model in the same way as before (with the fit() method) using attribute list 'algo'.
# 4): Print resulting matrix and name of the selected algorithm.

if best_algo == 'LogisticRegression':
	algo = getattr(sklearn.linear_model, best_algo)()

if  best_algo == 'SGDClassifier':
	algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

if (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
	algo = getattr(sklearn.ensemble, best_algo)()

if best_algo == 'SVC':
	algo = getattr(sklearn.svm, best_algo)(probability=True)

if best_algo == 'ComplementNB':
	algo = getattr(sklearn.naive_bayes, best_algo)()

if best_algo == 'MLPClassifier':
	algo = getattr(sklearn.neural_network, best_algo)()

if best_algo == 'XGBClassifier':
	algo = getattr(xgboost, best_algo)()

if best_algo == 'KNeighborsClassifier':
	algo = getattr(sklearn.neighbors, best_algo)()

if (best_algo == 'LinearDiscriminantAnalysis') or (best_algo == 'QuadraticDiscriminantAnalysis'):
	algo = getattr(sklearn.discriminant_analysis, best_algo)()

algo.fit(X_train, y_train)
name = algo.__class__.__name__

print("...remember, there are occasionally slight fluctuations in model performance on the same withheld samples...")
print("#"*30)
print(name)

# Code Block 14:
# 1): Recalling that 'algo' is a list of values of the selected algorithm's attributes (and thus should also be a
# sktest class object of whichever subclass the algorithm belongs to), conduct a series of method calls to derive
# other important statistics (AUC, Accuracy, Balanced Accuracy, Log Loss).
# Notes: splice notation may be used to select appropriate elements of any resulting matrices.
# All values formatted to 4 decimal places.

test_predictions = algo.predict_proba(X_test)
test_predictions = test_predictions[:, 1]
rocauc = roc_auc_score(y_test, test_predictions)
print("AUC: {:.4%}".format(rocauc))

test_predictions = algo.predict(X_test)
acc = accuracy_score(y_test, test_predictions)
print("Accuracy: {:.4%}".format(acc))

test_predictions = algo.predict(X_test)
balacc = balanced_accuracy_score(y_test, test_predictions)
print("Balanced Accuracy: {:.4%}".format(balacc))

test_predictions = algo.predict_proba(X_test)
ll = log_loss(y_test, test_predictions)
print("Log Loss: {:.4}".format(ll))

# Code Block 15: Save attribute list 'algo' (via dump() method) using imported joblib methods.
from joblib import dump, load
algo_out = prefix + '.trainedModel.joblib'
dump(algo, algo_out)

print("#"*30)
print("... this model has been saved as", algo_out, "for later use and can be found in your working directory.")

# Code Block 16: Now we'll export the AUC curve from the "data withheld" testing sample, turning it into a nice graph.
# 1): We use the pyplot library to convert saved model attributes into a MATLAB-like plot.
# 2): Create file handle 'plot_out' w/ prefix corresponding to selected algorithm - remember that the plot should
# NOT include the PHENO DataTable in 'y' as this is the 'withheld' graph.
# 3): Re-run the predict_proba() method on X_test and use the output to derive thresholds & 'roc_auc' via auc() method.
# 4): Use methods of the pyplot module to plot the graph, draw the ROC curve (in purple) to the graph,
# label it, limit the bounds, and finally save it as a figure. Print the resulting graph as well.

import matplotlib.pyplot as plt

plot_out = prefix + '.trainedModel_withheldSample_ROC.png'

test_predictions = algo.predict_proba(X_test)
test_predictions = test_predictions[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='purple', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic (ROC) - ' + best_algo )
plt.legend(loc="lower right")
plt.savefig(plot_out, dpi = 600)

print()
print("We are also exporting a ROC curve for you here", plot_out, "this is a graphical representation of AUC in the withheld test data for the best performing algorithm.")

# Code Block 17: We want to export predictions for the withheld sample separately from the test set, which is the obj.
# 1): Repeat the above process, this time creating separate predict_proba() outputs in 'test_predicteds_probs'
# (same as before) and 'test_predicted_cases' which uses the 'predict' method instead. Note the first method returns
# the chances of both negative and positive cases; the second just returns the selected outcome.
# 2): Create two new DataFrame objects 'test_case_probs_df' and 'test_predicted_cases_df' composed of the pos/neg
# probabilities returned by respective .predict_proba() and .predict() methods from the last step.
# 3): Create another two new DataFrame objects 'y_test_df' and 'IDs_test_df' using stored 'y_test' and 'IDs_test'.
# 4): Concat all four DataFrames, resetting index for 'y_test_df', 'test_case_probs_df', and 'test_predicted_cases_df'.
# In this way, only the index of the 'IDs_test_df' DataFrame is conserved.
# 5): Use the pd .columns() method to rename columns and the .drop() method to eliminate the 'INDEX' column.
# If I'm understanding right, the indexes of that column are still conserved.
# 6): Create appropriate file handle and export the final 'test_out' DataFrame via .to_csv() method.
# 7): Print a preview of the resulting file.

test_predicteds_probs = algo.predict_proba(X_test)
test_case_probs = test_predicteds_probs[:, 1]
test_predicted_cases = algo.predict(X_test)

test_case_probs_df = pd.DataFrame(test_case_probs)
test_predicted_cases_df = pd.DataFrame(test_predicted_cases)
y_test_df = pd.DataFrame(y_test)
IDs_test_df = pd.DataFrame(IDs_test)

test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_case_probs_df.reset_index(drop=True), test_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
test_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
test_out = test_out.drop(columns=['INDEX'])

test_outfile = prefix + '.trainedModel_withheldSample_Predictions.csv'
test_out.to_csv(test_outfile, index=False)

print("")
print("Preview of the exported predictions for the withheld test data that has been exported as", test_outfile, "these are pretty straight forward.")
print("They generally include the sample ID, the previously reported case status (1 = case), the case probability from the best performing algorithm and the predicted label from that algorithm,")
print("#"*30)
print(test_out.head())
print("#"*30)

# Code Block 18: Export the training data, which is by nature overfit.
# Repeat all the steps of Code Block 17, this time using the "trained" model's probabilities and associated DataFrames.

train_predicteds_probs = algo.predict_proba(X_train)
train_case_probs = train_predicteds_probs[:, 1]
train_predicted_cases = algo.predict(X_train)

train_case_probs_df = pd.DataFrame(train_case_probs)
train_predicted_cases_df = pd.DataFrame(train_predicted_cases)
y_train_df = pd.DataFrame(y_train)
IDs_train_df = pd.DataFrame(IDs_train)

train_out = pd.concat([IDs_train_df.reset_index(), y_train_df.reset_index(drop=True), train_case_probs_df.reset_index(drop=True), train_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
train_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
train_out = train_out.drop(columns=['INDEX'])

train_outfile = prefix + '.trainedModel_trainingSample_Predictions.csv'
train_out.to_csv(train_outfile, index=False)

print("")
print("Preview of the exported predictions for the training samples which is naturally overfit and exported as", train_outfile, "in the similar format as in the withheld test dataset that was just exported.")
print("#"*30)
print(train_out.head())
print("#"*30)

# Code Block 19: Export the histograms of probabilities via the seagram lib.

import seaborn as sns

genoML_colors = ["cyan","purple"]

g = sns.FacetGrid(train_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=False, rug=True))
g.add_legend()

plot_out = prefix + '.trainedModel_withheldSample_probabilities.png'
g.savefig(plot_out, dpi=600)

print("")
print("We are also exporting probability density plots to the file", plot_out, "this is a plot of the probability distributions of being a case, stratified by case and control status in the withheld test samples.")

## Export feature ranks.

print("")

feature_trigger = args.rank_features

if (feature_trigger == 'run'):

	if (best_algo == 'SVC') or (best_algo == 'ComplementNB') or (best_algo == 'KNeighborsClassifier') or (best_algo == 'QuadraticDiscriminantAnalysis') or (best_algo == 'BaggingClassifier'):
	
		print("Even if you selected to run feature ranking, you can't generate feature ranks using SVC, ComplementNB, KNeighborsClassifier, QuadraticDiscriminantAnalysis or BaggingClassifier... it just isn't possible.")
	
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

