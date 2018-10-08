pip install package_name
# or #
conda install package_name



pip install package_name
# or #
conda install package_name



import pandas as pd
import numpy as np



import pandas as pd
import numpy as np



student = pd.read_csv('/data/class.csv')



student = pd.read_csv('/data/class.csv')



# Notice you must specify the file location, as well as the name of the sheet 
# of the .xls file you want to import
student_xls = pd.read_excel(open('/data/class.xls', 'rb'),
                            sheetname='class')



# Notice you must specify the file location, as well as the name of the sheet 
# of the .xls file you want to import
student_xls = pd.read_excel(open('/data/class.xls', 'rb'),
                            sheetname='class')



student_json = pd.read_json('/data/class.json')



student_json = pd.read_json('/data/class.json')



print(student.shape)



print(student.shape)



print(student.info())



print(student.info())



print(student.head())



print(student.head())



print(student.mean())



print(student.mean())



print(student.describe())



print(student.describe())



# Notice the subsetting of student with [] and the name of the variable in 
# quotes ("")
print(student["Weight"].std())



# Notice the subsetting of student with [] and the name of the variable in 
# quotes ("")
print(student["Weight"].std())



print(student["Weight"].sum())



print(student["Weight"].sum())



print(student["Weight"].count())



print(student["Weight"].count())



print(student["Weight"].max())



print(student["Weight"].max())



print(student["Weight"].min())



print(student["Weight"].min())



print(student["Weight"].median())



print(student["Weight"].median())



# columns = "count" indicates to make the descriptive portion of the table 
# the counts of each level of the index variable
print(pd.crosstab(index=student["Age"], columns="count"))



# columns = "count" indicates to make the descriptive portion of the table 
# the counts of each level of the index variable
print(pd.crosstab(index=student["Age"], columns="count"))



print(pd.crosstab(index=student["Sex"], columns="count"))



print(pd.crosstab(index=student["Sex"], columns="count"))



# Notice the specification of a variable for the columns argument, instead 
# of "count"
print(pd.crosstab(index=student["Age"], columns=student["Sex"]))



# Notice the specification of a variable for the columns argument, instead 
# of "count"
print(pd.crosstab(index=student["Age"], columns=student["Sex"]))



females = student.query('Sex == "F"')
print(females.head())



females = student.query('Sex == "F"')
print(females.head())



# axis = 1 option indicates to concatenate column-wise
height_weight = pd.concat([student["Height"], student["Weight"]], axis = 1)
print(height_weight.corr(method = "pearson"))



# axis = 1 option indicates to concatenate column-wise
height_weight = pd.concat([student["Height"], student["Weight"]], axis = 1)
print(height_weight.corr(method = "pearson"))



import matplotlib.pyplot as plt



import matplotlib.pyplot as plt



plt.hist(student["Weight"], bins=[40,60,80,100,120,140,160])
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()



plt.hist(student["Weight"], bins=[40,60,80,100,120,140,160])
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()



# showmeans=True tells Python to plot the mean of the variable on the boxplot 
plt.boxplot(student["Weight"], showmeans=True)

# prevents Python from printing a "1" at the bottom of the boxplot
plt.xticks([])

plt.ylabel('Weight')
plt.show()



# showmeans=True tells Python to plot the mean of the variable on the boxplot 
plt.boxplot(student["Weight"], showmeans=True)

# prevents Python from printing a "1" at the bottom of the boxplot
plt.xticks([])

plt.ylabel('Weight')
plt.show()



# Notice here you specify the x variable, followed by the y variable 
plt.scatter(student["Height"], student["Weight"])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()



# Notice here you specify the x variable, followed by the y variable 
plt.scatter(student["Height"], student["Weight"])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()



x = student["Height"]
y = student["Weight"]

# np.polyfit() models Weight as a function of Height and returns the 
# parameters
m, b = np.polyfit(x, y, 1)
plt.scatter(x, y)

# plt.text() prints the equation of the line of best fit, with the first two 
# arguments specifying the x and y locations of the text, respectively 
# "%f" indicates to print a floating point number, that is specified following
# the string and a "%" character
plt.text(51, 140, "Line: y = %f x + %f"% (m,b))
plt.plot(x, m*x + b)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()



x = student["Height"]
y = student["Weight"]

# np.polyfit() models Weight as a function of Height and returns the 
# parameters
m, b = np.polyfit(x, y, 1)
plt.scatter(x, y)

# plt.text() prints the equation of the line of best fit, with the first two 
# arguments specifying the x and y locations of the text, respectively 
# "%f" indicates to print a floating point number, that is specified following
# the string and a "%" character
plt.text(51, 140, "Line: y = %f x + %f"% (m,b))
plt.plot(x, m*x + b)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()



# Get the counts of Sex 
counts = pd.crosstab(index=student["Sex"], columns="count")

# len() returns the number of categories of Sex (2)
# np.arange() creates a vector of the specified length
num = np.arange(len(counts))

# alpha = 0.5 changes the transparency of the bars
plt.bar(num, counts["count"], align='center', alpha=0.5)

# Set the xticks to be the indices of counts
plt.xticks(num, counts.index)
plt.xlabel("Sex")
plt.ylabel("Frequency")
plt.show()



# Get the counts of Sex 
counts = pd.crosstab(index=student["Sex"], columns="count")

# len() returns the number of categories of Sex (2)
# np.arange() creates a vector of the specified length
num = np.arange(len(counts))

# alpha = 0.5 changes the transparency of the bars
plt.bar(num, counts["count"], align='center', alpha=0.5)

# Set the xticks to be the indices of counts
plt.xticks(num, counts.index)
plt.xlabel("Sex")
plt.ylabel("Frequency")
plt.show()



# Subset data set to return only female weights, and then only male weights 
Weight_F = np.array(student.query('Sex == "F"')["Weight"])
Weight_M = np.array(student.query('Sex == "M"')["Weight"])
Weights = [Weight_F, Weight_M]

# PyPlot automatically plots the two weights side-by-side since Weights 
# is a 2D array
plt.boxplot(Weights, showmeans=True, labels=('F', 'M'))
plt.xlabel('Sex')
plt.ylabel('Weight')
plt.show()



# Subset data set to return only female weights, and then only male weights 
Weight_F = np.array(student.query('Sex == "F"')["Weight"])
Weight_M = np.array(student.query('Sex == "M"')["Weight"])
Weights = [Weight_F, Weight_M]

# PyPlot automatically plots the two weights side-by-side since Weights 
# is a 2D array
plt.boxplot(Weights, showmeans=True, labels=('F', 'M'))
plt.xlabel('Sex')
plt.ylabel('Weight')
plt.show()



import seaborn as sns
sns.boxplot(x="Sex", y="Weight", hue="Sex", data = student, showmeans=True)
sns.plt.show()



import seaborn as sns
sns.boxplot(x="Sex", y="Weight", hue="Sex", data = student, showmeans=True)
sns.plt.show()



# Notice here how you can create the BMI column in the data set 
# just by naming it 
student["BMI"] = student["Weight"] / student["Height"]**2 * 703
print(student.head())



# Notice here how you can create the BMI column in the data set 
# just by naming it 
student["BMI"] = student["Weight"] / student["Height"]**2 * 703
print(student.head())



# Notice the use of the np.where() function for a single condition 
student["BMI Class"] = np.where(student["BMI"] < 19.0, "Underweight",
                                "Healthy")
print(student.head())



# Notice the use of the np.where() function for a single condition 
student["BMI Class"] = np.where(student["BMI"] < 19.0, "Underweight",
                                "Healthy")
print(student.head())



student["LogWeight"] = np.log(student["Weight"])
student["ExpAge"] = np.exp(student["Age"])
student["SqrtHeight"] = np.sqrt(student["Height"])
student["BMI Neg"] = np.where(student["BMI"] < 19.0, -student["BMI"],
                              student["BMI"])
student["BMI Pos"] = np.abs(student["BMI Neg"])

# Create a Boolean variable
student["BMI Check"] = (student["BMI Pos"] == student["BMI"])



student["LogWeight"] = np.log(student["Weight"])
student["ExpAge"] = np.exp(student["Age"])
student["SqrtHeight"] = np.sqrt(student["Height"])
student["BMI Neg"] = np.where(student["BMI"] < 19.0, -student["BMI"],
                              student["BMI"])
student["BMI Pos"] = np.abs(student["BMI Neg"])

# Create a Boolean variable
student["BMI Check"] = (student["BMI Pos"] == student["BMI"])



# axis = 1 indicates to drop columns instead of rows
student = student.drop(["LogWeight", "ExpAge", "SqrtHeight", "BMI Neg",
                        "BMI Pos", "BMI Check"], axis = 1)
print(student.head())



# axis = 1 indicates to drop columns instead of rows
student = student.drop(["LogWeight", "ExpAge", "SqrtHeight", "BMI Neg",
                        "BMI Pos", "BMI Check"], axis = 1)
print(student.head())



# Notice kind="mergesort" which indicates to use a stable sorting 
# algorithm 
student = student.sort_values(by="Age", kind="mergesort")
print(student.head())



# Notice kind="mergesort" which indicates to use a stable sorting 
# algorithm 
student = student.sort_values(by="Age", kind="mergesort")
print(student.head())



student = student.sort_values(by="Sex", kind="mergesort")
# Notice that the data is now sorted first by Sex and then within Sex by Age 
print(student.head())



student = student.sort_values(by="Sex", kind="mergesort")
# Notice that the data is now sorted first by Sex and then within Sex by Age 
print(student.head())



print(student.groupby(by="Sex").mean())



print(student.groupby(by="Sex").mean())



# Look at the tail of the data currently
print(student.tail())



# Look at the tail of the data currently
print(student.tail())



student = student.append({'Name':'Jane', 'Sex':'F', 'Age':14, 'Height':56.3,
                          'Weight':77.0, 'BMI':17.077695,
                          'BMI Class': 'Underweight'},
                         ignore_index=True)

# Notice the change in the indices because of the ignore_index=True option 
# which allows for a Series, or one-dimensional DataFrame, to be appended 
# to an existing DataFrame 



student = student.append({'Name':'Jane', 'Sex':'F', 'Age':14, 'Height':56.3,
                          'Weight':77.0, 'BMI':17.077695,
                          'BMI Class': 'Underweight'},
                         ignore_index=True)

# Notice the change in the indices because of the ignore_index=True option 
# which allows for a Series, or one-dimensional DataFrame, to be appended 
# to an existing DataFrame 



def toKG(lb):
    return (0.45359237 * lb)

student["Weight KG"] = student["Weight"].apply(toKG)
print(student.head())



def toKG(lb):
    return (0.45359237 * lb)

student["Weight KG"] = student["Weight"].apply(toKG)
print(student.head())



# Notice the use of the fish data set because it has some missing 
# observations 
fish = pd.read_csv('/data/fish.csv')

# First sort by Weight, requesting those with NA for Weight first 
fish = fish.sort_values(by='Weight', kind='mergesort', na_position='first')
print(fish.head())



# Notice the use of the fish data set because it has some missing 
# observations 
fish = pd.read_csv('/data/fish.csv')

# First sort by Weight, requesting those with NA for Weight first 
fish = fish.sort_values(by='Weight', kind='mergesort', na_position='first')
print(fish.head())



new_fish = fish.dropna()
print(new_fish.head())



new_fish = fish.dropna()
print(new_fish.head())



# Notice the use of the student data set again, however we want to reload it
# without the changes we've made previously 
student = pd.read_csv('/data/class.csv')
student1 = pd.concat([student["Name"], student["Sex"], student["Age"]],
                    axis = 1)
print(student1.head())



# Notice the use of the student data set again, however we want to reload it
# without the changes we've made previously 
student = pd.read_csv('/data/class.csv')
student1 = pd.concat([student["Name"], student["Sex"], student["Age"]],
                    axis = 1)
print(student1.head())



student2 = pd.concat([student["Name"], student["Height"], student["Weight"]],
                    axis = 1)
print(student2.head())



student2 = pd.concat([student["Name"], student["Height"], student["Weight"]],
                    axis = 1)
print(student2.head())



new = pd.merge(student1, student2, on="Name")
print(new.head())



new = pd.merge(student1, student2, on="Name")
print(new.head())



print(student.equals(new))



print(student.equals(new))



newstudent1 = pd.concat([student["Name"], student["Sex"], student["Age"]],
                        axis = 1)
print(newstudent1.head())



newstudent1 = pd.concat([student["Name"], student["Sex"], student["Age"]],
                        axis = 1)
print(newstudent1.head())



newstudent2 = pd.concat([student["Height"], student["Weight"]], axis = 1)
print(newstudent2.head())



newstudent2 = pd.concat([student["Height"], student["Weight"]], axis = 1)
print(newstudent2.head())



new2 = newstudent1.join(newstudent2)
print(new2.head())



new2 = newstudent1.join(newstudent2)
print(new2.head())



print(student.equals(new2))



print(student.equals(new2))



# Notice we are using a new data set that needs to be read into the 
# environment 
price = pd.read_csv('/data/price.csv')

# The following code is used to remove the "," and "$" characters from 
# the ACTUAL colum so that the values can be summed 
from re import sub
from decimal import Decimal
def trim_money(money):
    return(float(Decimal(sub(r'[^\d.]', '', money))))

price["REVENUE"] = price["ACTUAL"].apply(trim_money)
table = pd.pivot_table(price, index=["COUNTRY", "STATE", "PRODTYPE",
                                     "PRODUCT"], values="REVENUE",
                       aggfunc=np.sum)
print(table.head())



# Notice we are using a new data set that needs to be read into the 
# environment 
price = pd.read_csv('/data/price.csv')

# The following code is used to remove the "," and "$" characters from 
# the ACTUAL colum so that the values can be summed 
from re import sub
from decimal import Decimal
def trim_money(money):
    return(float(Decimal(sub(r'[^\d.]', '', money))))

price["REVENUE"] = price["ACTUAL"].apply(trim_money)
table = pd.pivot_table(price, index=["COUNTRY", "STATE", "PRODTYPE",
                                     "PRODUCT"], values="REVENUE",
                       aggfunc=np.sum)
print(table.head())



print(np.unique(price["STATE"]))



print(np.unique(price["STATE"]))



# Notice we are using a new data set that needs to be read into the 
# environment 
iris = pd.read_csv('/data/iris.csv')
features = iris.drop(["Target"], axis = 1)

from sklearn import preprocessing
features_scaled = preprocessing.scale(features.as_matrix())

from sklearn.decomposition import PCA

pca = PCA(n_components = 4)
pca = pca.fit(features_scaled)
print(np.transpose(pca.components_))



# Notice we are using a new data set that needs to be read into the 
# environment 
iris = pd.read_csv('/data/iris.csv')
features = iris.drop(["Target"], axis = 1)

from sklearn import preprocessing
features_scaled = preprocessing.scale(features.as_matrix())

from sklearn.decomposition import PCA

pca = PCA(n_components = 4)
pca = pca.fit(features_scaled)
print(np.transpose(pca.components_))



from sklearn.model_selection import train_test_split

target = iris["Target"]

# The following code splits the iris data set into 70% train and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    test_size = 0.3,
                                                    random_state = 29)
train_x = pd.DataFrame(X_train)
train_y = pd.DataFrame(Y_train)
test_x = pd.DataFrame(X_test)
test_y = pd.DataFrame(Y_test)

train = pd.concat([train_x, train_y], axis = 1)
test = pd.concat([test_x, test_y], axis = 1)

train.to_csv('/data/iris_train_Python.csv', index = False)
test.to_csv('/data/iris_test_Python.csv', index = False)



from sklearn.model_selection import train_test_split

target = iris["Target"]

# The following code splits the iris data set into 70% train and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                    test_size = 0.3,
                                                    random_state = 29)
train_x = pd.DataFrame(X_train)
train_y = pd.DataFrame(Y_train)
test_x = pd.DataFrame(X_test)
test_y = pd.DataFrame(Y_test)

train = pd.concat([train_x, train_y], axis = 1)
test = pd.concat([test_x, test_y], axis = 1)

train.to_csv('/data/iris_train_Python.csv', index = False)
test.to_csv('/data/iris_test_Python.csv', index = False)



# Notice we are using a new data set that needs to be read into the 
# environment 
tips = pd.read_csv('/data/tips.csv')

# The following code is used to determine if the individual left more 
# than a 15% tip 
tips["fifteen"] = 0.15 * tips["total_bill"]
tips["greater15"] = np.where(tips["tip"] > tips["fifteen"], 1, 0)

import statsmodels.api as sm

# Notice the syntax of greater15 as a function of total_bill 
logreg = sm.formula.glm("greater15 ~ total_bill",
                       family=sm.families.Binomial(),
                       data=tips).fit()
print(logreg.summary())



# Notice we are using a new data set that needs to be read into the 
# environment 
tips = pd.read_csv('/data/tips.csv')

# The following code is used to determine if the individual left more 
# than a 15% tip 
tips["fifteen"] = 0.15 * tips["total_bill"]
tips["greater15"] = np.where(tips["tip"] > tips["fifteen"], 1, 0)

import statsmodels.api as sm

# Notice the syntax of greater15 as a function of total_bill 
logreg = sm.formula.glm("greater15 ~ total_bill",
                       family=sm.families.Binomial(),
                       data=tips).fit()
print(logreg.summary())



# Fit a linear regression model of tip by total_bill
from sklearn.linear_model import LinearRegression

# If your data has one feature, you need to reshape the 1D array
linreg = LinearRegression()
linreg.fit(tips["total_bill"].values.reshape(-1,1), tips["tip"])
print(linreg.coef_)
print(linreg.intercept_)



# Fit a linear regression model of tip by total_bill
from sklearn.linear_model import LinearRegression

# If your data has one feature, you need to reshape the 1D array
linreg = LinearRegression()
linreg.fit(tips["total_bill"].values.reshape(-1,1), tips["tip"])
print(linreg.coef_)
print(linreg.intercept_)



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/tips_train.csv')
test = pd.read_csv('/data/tips_test.csv')

train["fifteen"] = 0.15 * train["total_bill"]
train["greater15"] = np.where(train["tip"] > train["fifteen"], 1, 0)
test["fifteen"] = 0.15 * test["total_bill"]
test["greater15"] = np.where(test["tip"] > test["fifteen"], 1, 0)

logreg = sm.formula.glm("greater15 ~ total_bill",
                       family=sm.families.Binomial(),
                       data=train).fit()
print(logreg.summary())



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/tips_train.csv')
test = pd.read_csv('/data/tips_test.csv')

train["fifteen"] = 0.15 * train["total_bill"]
train["greater15"] = np.where(train["tip"] > train["fifteen"], 1, 0)
test["fifteen"] = 0.15 * test["total_bill"]
test["greater15"] = np.where(test["tip"] > test["fifteen"], 1, 0)

logreg = sm.formula.glm("greater15 ~ total_bill",
                       family=sm.families.Binomial(),
                       data=train).fit()
print(logreg.summary())



# Prediction on testing data
predictions = logreg.predict(test["total_bill"])
predY = np.where(predictions < 0.5, 0, 1)

# If the prediction probability is less than 0.5, classify this as a 0
# and otherwise classify as a 1.  This isn't the best method -- a better 
# method would be randomly assigning a 0 or 1 when a probability of 0.5 
# occurrs, but this insures that results are consistent 

# Determine how many were correctly classified 
Results = np.where(predY == test["greater15"], "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
predictions = logreg.predict(test["total_bill"])
predY = np.where(predictions < 0.5, 0, 1)

# If the prediction probability is less than 0.5, classify this as a 0
# and otherwise classify as a 1.  This isn't the best method -- a better 
# method would be randomly assigning a 0 or 1 when a probability of 0.5 
# occurrs, but this insures that results are consistent 

# Determine how many were correctly classified 
Results = np.where(predY == test["greater15"], "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Fit a linear regression model
linreg = LinearRegression()
linreg.fit(train.drop(["Target"], axis = 1), train["Target"])
print(linreg.coef_)
print(linreg.intercept_)



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Fit a linear regression model
linreg = LinearRegression()
linreg.fit(train.drop(["Target"], axis = 1), train["Target"])
print(linreg.coef_)
print(linreg.intercept_)



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = linreg.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (prediction["predY"] - test["Target"])**2
print(np.mean(prediction["sq_diff"]))



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = linreg.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (prediction["predY"] - test["Target"])**2
print(np.mean(prediction["sq_diff"]))



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.tree import DecisionTreeClassifier

# random_state is used to specify a seed for a random integer so that the 
# results are reproducible
treeMod = DecisionTreeClassifier(random_state=29)
treeMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = treeMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Notice we are using new data sets that need to be read into the environment 
train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.tree import DecisionTreeClassifier

# random_state is used to specify a seed for a random integer so that the 
# results are reproducible
treeMod = DecisionTreeClassifier(random_state=29)
treeMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = treeMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
predY = treeMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
predY = treeMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.tree import DecisionTreeRegressor

treeMod = DecisionTreeRegressor(random_state=29)
treeMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = treeMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.tree import DecisionTreeRegressor

treeMod = DecisionTreeRegressor(random_state=29)
treeMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = treeMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = treeMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (prediction["predY"] - test["Target"])**2
print(np.mean(prediction["sq_diff"]))



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = treeMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (prediction["predY"] - test["Target"])**2
print(np.mean(prediction["sq_diff"]))



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.ensemble import RandomForestClassifier

rfMod = RandomForestClassifier(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = rfMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.ensemble import RandomForestClassifier

rfMod = RandomForestClassifier(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = rfMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.ensemble import RandomForestRegressor

rfMod = RandomForestRegressor(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = rfMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.ensemble import RandomForestRegressor

rfMod = RandomForestRegressor(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = rfMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.ensemble import GradientBoostingClassifier

# n_estimators = total number of trees to fit which is analogous to the 
# number of iterations
# learning_rate = shrinkage or step-size reduction, where a lower 
# learning rate requires more iterations
gbMod = GradientBoostingClassifier(random_state = 29, learning_rate = .01,
                                  n_estimators = 2500)
gbMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = gbMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from sklearn.ensemble import GradientBoostingClassifier

# n_estimators = total number of trees to fit which is analogous to the 
# number of iterations
# learning_rate = shrinkage or step-size reduction, where a lower 
# learning rate requires more iterations
gbMod = GradientBoostingClassifier(random_state = 29, learning_rate = .01,
                                  n_estimators = 2500)
gbMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = gbMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
predY = gbMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
predY = gbMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.ensemble import GradientBoostingRegressor

gbMod = GradientBoostingRegressor(random_state = 29, learning_rate = .01,
                                  n_estimators = 2500)
gbMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = gbMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from sklearn.ensemble import GradientBoostingRegressor

gbMod = GradientBoostingRegressor(random_state = 29, learning_rate = .01,
                                  n_estimators = 2500)
gbMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Determine variable importance
var_import = gbMod.feature_importances_
var_import = pd.DataFrame(var_import)
var_import = var_import.rename(columns = {0:'Importance'})
var_import = var_import.sort_values(by="Importance", kind = "mergesort",
                                    ascending = False)
print(var_import.head())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = gbMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = gbMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from xgboost import XGBClassifier

# Fit XGBClassifier model on training data
xgbMod = XGBClassifier(seed = 29, learning_rate = 0.01,
                       n_estimators = 2500)
xgbMod.fit(train.drop(["Target"], axis = 1), train["Target"])



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

from xgboost import XGBClassifier

# Fit XGBClassifier model on training data
xgbMod = XGBClassifier(seed = 29, learning_rate = 0.01,
                       n_estimators = 2500)
xgbMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Prediction on testing data
predY = xgbMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
predY = xgbMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == predY, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from xgboost import XGBRegressor

# Fit XGBRegressor model on training data
xgbMod = XGBRegressor(seed = 29, learning_rate = 0.01,
                      n_estimators = 2500)
xgbMod.fit(train.drop(["Target"], axis = 1), train["Target"])



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

from xgboost import XGBRegressor

# Fit XGBRegressor model on training data
xgbMod = XGBRegressor(seed = 29, learning_rate = 0.01,
                      n_estimators = 2500)
xgbMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = xgbMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = xgbMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

# Fit a support vector classification model
from sklearn.svm import SVC
svMod = SVC(random_state = 29, kernel = 'linear')
svMod.fit(train.drop(["Target"], axis = 1), train["Target"])



train = pd.read_csv('/data/breastcancer_train.csv')
test = pd.read_csv('/data/breastcancer_test.csv')

# Fit a support vector classification model
from sklearn.svm import SVC
svMod = SVC(random_state = 29, kernel = 'linear')
svMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Prediction on testing data
prediction = svMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == prediction, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



# Prediction on testing data
prediction = svMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified
Results = np.where(test["Target"] == prediction, "Correct", "Wrong")
print(pd.crosstab(index=Results, columns="count"))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Fit a support vector regression model
from sklearn.svm import SVR
svMod = SVR()
svMod.fit(train.drop(["Target"], axis = 1), train["Target"])



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Fit a support vector regression model
from sklearn.svm import SVR
svMod = SVR()
svMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = svMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



# Prediction on testing data
prediction = pd.DataFrame()
prediction["predY"] = svMod.predict(test.drop(["Target"], axis = 1))

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



# Notice we are using new data sets
train = pd.read_csv('/data/digits_train.csv')
test = pd.read_csv('/data/digits_test.csv')

# Fit a neural network classification model on training data
from sklearn.neural_network import MLPClassifier
nnMod = MLPClassifier(max_iter = 200, hidden_layer_sizes=(100,),
                      random_state = 29)
nnMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Notice we are using new data sets
train = pd.read_csv('/data/digits_train.csv')
test = pd.read_csv('/data/digits_test.csv')

# Fit a neural network classification model on training data
from sklearn.neural_network import MLPClassifier
nnMod = MLPClassifier(max_iter = 200, hidden_layer_sizes=(100,),
                      random_state = 29)
nnMod.fit(train.drop(["Target"], axis = 1), train["Target"])



# Prediction on testing data
predY = nnMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified

from sklearn.metrics import confusion_matrix

print(confusion_matrix(test["Target"], predY))



# Prediction on testing data
predY = nnMod.predict(test.drop(["Target"], axis = 1))

# Determine how many were correctly classified

from sklearn.metrics import confusion_matrix

print(confusion_matrix(test["Target"], predY))



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Scale input data
from sklearn.preprocessing import StandardScaler

train_features = train.drop(["Target"], axis = 1)
scaler = StandardScaler().fit(np.array(train_features))
train_scaled = scaler.transform(np.array(train_features))
train_scaled = pd.DataFrame(train_scaled)

test_features = test.drop(["Target"], axis = 1)
scaler = StandardScaler().fit(np.array(test_features))
test_scaled = scaler.transform(np.array(test_features))
test_scaled = pd.DataFrame(test_scaled)

# Fit neural network regression model, dividing target by 50 for scaling
from sklearn.neural_network import MLPRegressor
nnMod = MLPRegressor(max_iter = 250, random_state = 29, solver = 'lbfgs')
nnMod = nnMod.fit(train_scaled, train["Target"] / 50)



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Scale input data
from sklearn.preprocessing import StandardScaler

train_features = train.drop(["Target"], axis = 1)
scaler = StandardScaler().fit(np.array(train_features))
train_scaled = scaler.transform(np.array(train_features))
train_scaled = pd.DataFrame(train_scaled)

test_features = test.drop(["Target"], axis = 1)
scaler = StandardScaler().fit(np.array(test_features))
test_scaled = scaler.transform(np.array(test_features))
test_scaled = pd.DataFrame(test_scaled)

# Fit neural network regression model, dividing target by 50 for scaling
from sklearn.neural_network import MLPRegressor
nnMod = MLPRegressor(max_iter = 250, random_state = 29, solver = 'lbfgs')
nnMod = nnMod.fit(train_scaled, train["Target"] / 50)



# Prediction on testing data, remembering to multiply by 50
prediction = pd.DataFrame()
prediction["predY"] = nnMod.predict(test_scaled)*50

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



# Prediction on testing data, remembering to multiply by 50
prediction = pd.DataFrame()
prediction["predY"] = nnMod.predict(test_scaled)*50

# Determine mean squared error
prediction["sq_diff"] = (test["Target"] - prediction["predY"])**2
print(prediction["sq_diff"].mean())



iris = pd.read_csv('/data/iris.csv')
iris["Species"] = np.where(iris["Target"] == 0, "Setosa",
                           np.where(iris["Target"] == 1, "Versicolor",
                                    "Virginica"))
features = pd.concat([iris["PetalLength"], iris["PetalWidth"],
                     iris["SepalLength"], iris["SepalWidth"]], axis = 1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, random_state = 29).fit(features)

print(pd.crosstab(index = iris["Species"], columns = kmeans.labels_))



iris = pd.read_csv('/data/iris.csv')
iris["Species"] = np.where(iris["Target"] == 0, "Setosa",
                           np.where(iris["Target"] == 1, "Versicolor",
                                    "Virginica"))
features = pd.concat([iris["PetalLength"], iris["PetalWidth"],
                     iris["SepalLength"], iris["SepalWidth"]], axis = 1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, random_state = 29).fit(features)

print(pd.crosstab(index = iris["Species"], columns = kmeans.labels_))



from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters = 3,
                              random_state = 29).fit(features)

print(pd.crosstab(index = iris["Species"], columns = spectral.labels_))



from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters = 3,
                              random_state = 29).fit(features)

print(pd.crosstab(index = iris["Species"], columns = spectral.labels_))



from sklearn.cluster import AgglomerativeClustering

aggl = AgglomerativeClustering(n_clusters = 3).fit(features)

print(pd.crosstab(index = iris["Species"], columns = aggl.labels_))



from sklearn.cluster import AgglomerativeClustering

aggl = AgglomerativeClustering(n_clusters = 3).fit(features)

print(pd.crosstab(index = iris["Species"], columns = aggl.labels_))



from sklearn.cluster import DBSCAN

dbscan = DBSCAN().fit(features)

print(pd.crosstab(index = iris["Species"], columns = dbscan.labels_))



from sklearn.cluster import DBSCAN

dbscan = DBSCAN().fit(features)

print(pd.crosstab(index = iris["Species"], columns = dbscan.labels_))



from pyclustering.nnet import som

sm = som.som(4,4)

sm.train(features.as_matrix(), 100)

sm.show_distance_matrix()



from pyclustering.nnet import som

sm = som.som(4,4)

sm.train(features.as_matrix(), 100)

sm.show_distance_matrix()



# Read in new data set
air = pd.read_csv('/data/air.csv')

air["DATE"] = pd.to_datetime(air["DATE"], infer_datetime_format = True)
air.index = air["DATE"].values

plt.plot(air.index, air["AIR"])
plt.show()



# Read in new data set
air = pd.read_csv('/data/air.csv')

air["DATE"] = pd.to_datetime(air["DATE"], infer_datetime_format = True)
air.index = air["DATE"].values

plt.plot(air.index, air["AIR"])
plt.show()



import pyflux as pf

# ar = 12 is necessary to indicate the seasonality of data
model = pf.ARIMA(data = air, ar = 12, ma = 1, integ = 0, target = 'AIR',
                 family = pf.Normal())
x = model.fit("MLE")
model.plot_predict(h = 24)



import pyflux as pf

# ar = 12 is necessary to indicate the seasonality of data
model = pf.ARIMA(data = air, ar = 12, ma = 1, integ = 0, target = 'AIR',
                 family = pf.Normal())
x = model.fit("MLE")
model.plot_predict(h = 24)



# Read in new data set
usecon = pd.read_csv('/data/usecon.csv')

petrol = usecon["PETROL"]

plt.plot(petrol)
plt.show()



# Read in new data set
usecon = pd.read_csv('/data/usecon.csv')

petrol = usecon["PETROL"]

plt.plot(petrol)
plt.show()



def simple_exp_smoothing(data, alpha, n_preds):
    # Eq1:
    output = [data[0]]
    # Smooth given data plus we want to predict 24 units 
    # past the end
    for i in range(1, len(data) + n_preds):
        # Eq2:
        if (i < len(data)):
            output.append(alpha * data[i] + (1 - alpha) * data[i-1])
        else:
            output.append(alpha * output[i-1] + (1 - alpha) * output[i-2])
    return output

pred = simple_exp_smoothing(petrol, 0.9999, 24)

plt.plot(pd.DataFrame(pred), color = "red")
plt.plot(petrol, color = "blue")
plt.show()



def simple_exp_smoothing(data, alpha, n_preds):
    # Eq1:
    output = [data[0]]
    # Smooth given data plus we want to predict 24 units 
    # past the end
    for i in range(1, len(data) + n_preds):
        # Eq2:
        if (i < len(data)):
            output.append(alpha * data[i] + (1 - alpha) * data[i-1])
        else:
            output.append(alpha * output[i-1] + (1 - alpha) * output[i-2])
    return output

pred = simple_exp_smoothing(petrol, 0.9999, 24)

plt.plot(pd.DataFrame(pred), color = "red")
plt.plot(petrol, color = "blue")
plt.show()



vehicle = usecon["VEHICLE"]

plt.plot(vehicle)
plt.show()



vehicle = usecon["VEHICLE"]

plt.plot(vehicle)
plt.show()



def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals
def triple_exponential_smoothing_add(series, slen, alpha, beta, gamma,
                                     n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen])
                                                 + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) +
                                (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

add_preds = triple_exponential_smoothing_add(vehicles, 12, 0.5731265, 0,
                                             0.7230956, 24)

plt.plot(pd.DataFrame(add_preds), color = "red")
plt.plot(vehicles, color = "blue")
plt.show()



def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals
def triple_exponential_smoothing_add(series, slen, alpha, beta, gamma,
                                     n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen])
                                                 + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) +
                                (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

add_preds = triple_exponential_smoothing_add(vehicles, 12, 0.5731265, 0,
                                             0.7230956, 24)

plt.plot(pd.DataFrame(add_preds), color = "red")
plt.plot(vehicles, color = "blue")
plt.show()



from fbprophet import Prophet

air = pd.read_csv("/data/air.csv")

air_df = pd.DataFrame()

air_df["ds"] = pd.to_datetime(air["DATE"], infer_datetime_format = True)
air_df["y"] = air["AIR"]

m = Prophet(yearly_seasonality = True, weekly_seasonality = False)

m.fit(air_df)

future = m.make_future_dataframe(periods = 24, freq = "M")

forecast = m.predict(future)

m.plot(forecast)



from fbprophet import Prophet

air = pd.read_csv("/data/air.csv")

air_df = pd.DataFrame()

air_df["ds"] = pd.to_datetime(air["DATE"], infer_datetime_format = True)
air_df["y"] = air["AIR"]

m = Prophet(yearly_seasonality = True, weekly_seasonality = False)

m.fit(air_df)

future = m.make_future_dataframe(periods = 24, freq = "M")

forecast = m.predict(future)

m.plot(forecast)



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
rfMod = RandomForestRegressor(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Evaluation on training data
predY = rfMod.predict(train.drop(["Target"], axis = 1))

# Determine coefficient of determination score
from sklearn.metrics import r2_score
r2_rf = r2_score(train["Target"], predY)
print("Random forest regression model r^2 score (coefficient of determination): %f" % r2_rf)



train = pd.read_csv('/data/boston_train.csv')
test = pd.read_csv('/data/boston_test.csv')

# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
rfMod = RandomForestRegressor(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Evaluation on training data
predY = rfMod.predict(train.drop(["Target"], axis = 1))

# Determine coefficient of determination score
from sklearn.metrics import r2_score
r2_rf = r2_score(train["Target"], predY)
print("Random forest regression model r^2 score (coefficient of determination): %f" % r2_rf)



# Random Forest Regression Model (rfMod)

# Evaluation on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine coefficient of determination score
r2_rf = r2_score(test["Target"], predY)
print("Random forest regression model r^2 score (coefficient of determination): %f" % r2_rf)



# Random Forest Regression Model (rfMod)

# Evaluation on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine coefficient of determination score
r2_rf = r2_score(test["Target"], predY)
print("Random forest regression model r^2 score (coefficient of determination): %f" % r2_rf)



train = pd.read_csv('/data/digits_train.csv')
test = pd.read_csv('/data/digits_test.csv')

# Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
rfMod = RandomForestClassifier(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Evaluation on training data
predY = rfMod.predict(train.drop(["Target"], axis = 1))

# Determine accuracy score
from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(train["Target"], predY)
print("Random forest model accuracy: %f" % accuracy_rf)



train = pd.read_csv('/data/digits_train.csv')
test = pd.read_csv('/data/digits_test.csv')

# Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
rfMod = RandomForestClassifier(random_state=29)
rfMod.fit(train.drop(["Target"], axis = 1), train["Target"])

# Evaluation on training data
predY = rfMod.predict(train.drop(["Target"], axis = 1))

# Determine accuracy score
from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(train["Target"], predY)
print("Random forest model accuracy: %f" % accuracy_rf)



# Random Forest Classification Model (rfMod)

# Evaluation on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine accuracy score
accuracy_rf = accuracy_score(test["Target"], predY)
print("Random forest model accuracy: %f" % accuracy_rf)



# Random Forest Classification Model (rfMod)

# Evaluation on testing data
predY = rfMod.predict(test.drop(["Target"], axis = 1))

# Determine accuracy score
accuracy_rf = accuracy_score(test["Target"], predY)
print("Random forest model accuracy: %f" % accuracy_rf)



import numpy as np
my_array = np.array([1, 2, 3, 4])
print(my_array)



import numpy as np
my_array = np.array([1, 2, 3, 4])
print(my_array)



print(my_array[3])



print(my_array[3])



import pandas as pd
student = pd.read_csv('/data/class.csv')
for_dict = pd.concat([student["Name"], student["Age"]], axis = 1)
class_dict = for_dict.set_index('Name').T.to_dict('list')
print(class_dict.get('James'))



import pandas as pd
student = pd.read_csv('/data/class.csv')
for_dict = pd.concat([student["Name"], student["Age"]], axis = 1)
class_dict = for_dict.set_index('Name').T.to_dict('list')
print(class_dict.get('James'))



list1 = ['item1', 102]
print(list1)



list1 = ['item1', 102]
print(list1)



print(list1[1])



print(list1[1])



import pandas as pd
my_array = pd.Series([1, 3, 5, 9])
print(my_array)



import pandas as pd
my_array = pd.Series([1, 3, 5, 9])
print(my_array)



print(my_array[1])



print(my_array[1])



s = set(["1", "2", "3"])
# s is a set, which means you can add or delete elements from s
print(s)



s = set(["1", "2", "3"])
# s is a set, which means you can add or delete elements from s
print(s)



s.add("4")
print(s)



s.add("4")
print(s)



fs = frozenset(["1", "2", "3"])
# fs is a frozenset, which means you cannot add or delete elements from fs
print(fs)



fs = frozenset(["1", "2", "3"])
# fs is a frozenset, which means you cannot add or delete elements from fs
print(fs)



s = 'My first string!'
print(s)



s = 'My first string!'
print(s)



print(s[5])



print(s[5])



