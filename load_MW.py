import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
dataset = pandas.read_excel(r'C:\Users\Gabe Noble\Documents\ECEN 404\Scaled_MW_Loads.xlsx')
names = ['Date', 'Time', '1 Gen MW', '20 Gen MW', '30 Gen MW', '33 Gen MW', '43 Gen MW',
         '1 Gen MVar', '20 Gen MVar', '30 Gen MVar', '33 Gen MVar', '43 Gen MVar']

# shape
# print(dataset.shape)

# head
# print(dataset.head(20))

# descriptions
# print(dataset.describe())

# class distribution
# print(dataset.groupby('Time').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
# plt.show()

# histograms
# dataset.hist()
# plt.show()

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# Split-out validation dataset
# array = dataset.values
# X = array[:, 0:11]
# Y = array[:, 2]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                random_state=seed)

# Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'

# Spot Check Algorithms
# models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
#          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
#          ('SVM', SVC(gamma='auto'))]
# evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
#    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#    results.append(cv_results)
#    names.append(name)
#    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#    print(msg)

# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
