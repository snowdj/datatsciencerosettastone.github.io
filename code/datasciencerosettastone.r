install.packages("name_of_package")



install.packages("name_of_package")



student <- read.csv('/Users/class.csv')



student <- read.csv('/Users/class.csv')



# call the gdata package
library(gdata)

student_xls <- read.xls('/Users/class.xls', 1)



# call the gdata package
library(gdata)

student_xls <- read.xls('/Users/class.xls', 1)



# call the rjson package
library(rjson)

temp <- fromJSON(file = '/Users/class.json')
temp <- do.call('rbind', temp)
temp <- data.frame(temp, stringsAsFactors = TRUE)
temp <- transform(temp, Name=unlist(Name), Sex=unlist(Sex), Age=unlist(Age),
                  Height=unlist(Height), Weight=unlist(Weight))
temp$Name <- as.factor(temp$Name)
temp$Sex <- as.factor(temp$Sex)
temp$Age <- as.integer(temp$Age)

student_json <- temp



# call the rjson package
library(rjson)

temp <- fromJSON(file = '/Users/class.json')
temp <- do.call('rbind', temp)
temp <- data.frame(temp, stringsAsFactors = TRUE)
temp <- transform(temp, Name=unlist(Name), Sex=unlist(Sex), Age=unlist(Age),
                  Height=unlist(Height), Weight=unlist(Weight))
temp$Name <- as.factor(temp$Name)
temp$Sex <- as.factor(temp$Sex)
temp$Age <- as.integer(temp$Age)

student_json <- temp



dim(student)



dim(student)



str(student)



str(student)



head(student, n=5)



head(student, n=5)



# We must apply the is.numeric() function to the data set which returns a 
# matrix of booleans that we then use to subset the data set to return 
# only numeric variables  

# Then we can use the colMeans() function to return the means of 
# column variables
colMeans(student[sapply(student, is.numeric)])



# We must apply the is.numeric() function to the data set which returns a 
# matrix of booleans that we then use to subset the data set to return 
# only numeric variables  

# Then we can use the colMeans() function to return the means of 
# column variables
colMeans(student[sapply(student, is.numeric)])



summary(student)



summary(student)



# Notice the subsetting of student with the "$" character 
sd(student$Weight)



# Notice the subsetting of student with the "$" character 
sd(student$Weight)



sum(student$Weight)



sum(student$Weight)



length(student$Weight)



length(student$Weight)



max(student$Weight)



max(student$Weight)



min(student$Weight)



min(student$Weight)



median(student$Weight)



median(student$Weight)



table(student$Age)



table(student$Age)



table(student$Sex)



table(student$Sex)



table(student$Age, student$Sex)



table(student$Age, student$Sex)



# The "," character tells R to select all columns of the data set
females <- student[which(student$Sex == 'F'), ]
head(females, n=5)



# The "," character tells R to select all columns of the data set
females <- student[which(student$Sex == 'F'), ]
head(females, n=5)



height_weight <- subset(student, select = c(Height, Weight))
cor(height_weight, method = "pearson")



height_weight <- subset(student, select = c(Height, Weight))
cor(height_weight, method = "pearson")



Weight <- student$Weight
hist(Weight)



Weight <- student$Weight
hist(Weight)



# points(mean(Weight)) tells R to plot the mean on the boxplot 
boxplot(Weight, ylab="Weight")
points(mean(Weight))



# points(mean(Weight)) tells R to plot the mean on the boxplot 
boxplot(Weight, ylab="Weight")
points(mean(Weight))



Height <- student$Height
# Notice here you specify the x variable, followed by the y variable 
plot(Height, Weight)



Height <- student$Height
# Notice here you specify the x variable, followed by the y variable 
plot(Height, Weight)



plot(Height, Weight)

# lm() models Weight as a function of Height and returns the parameters 
# of the line of best fit
model <- lm(Weight~Height)
coeff <- coef(model)
intercept <- as.matrix(coeff[1])[1]
slope <- as.matrix(coeff[2])[1]

# abline() prints the line of best fit 
abline(lm(Weight~Height))

# text() prints the equation of the line of best fit, with the first 
# two arguments specifying the x and y location, respectively, of where 
# the text should be printed on the graph 
text(55, 140, bquote(Line: y == .(slope) * x + .(intercept)))



plot(Height, Weight)

# lm() models Weight as a function of Height and returns the parameters 
# of the line of best fit
model <- lm(Weight~Height)
coeff <- coef(model)
intercept <- as.matrix(coeff[1])[1]
slope <- as.matrix(coeff[2])[1]

# abline() prints the line of best fit 
abline(lm(Weight~Height))

# text() prints the equation of the line of best fit, with the first 
# two arguments specifying the x and y location, respectively, of where 
# the text should be printed on the graph 
text(55, 140, bquote(Line: y == .(slope) * x + .(intercept)))



counts <- table(student$Sex)

# beside = TRUE indicates to print the bars side by side instead of on top of 
# each other 
# names.arg indicates which names to use to label the bars 
barplot(counts, beside=TRUE, ylab= "Frequency", xlab= "Sex",
        names.arg=names(counts))



counts <- table(student$Sex)

# beside = TRUE indicates to print the bars side by side instead of on top of 
# each other 
# names.arg indicates which names to use to label the bars 
barplot(counts, beside=TRUE, ylab= "Frequency", xlab= "Sex",
        names.arg=names(counts))



# Subset data set to return only female weights, and then only male weights 
Female_Weight <- student[which(student$Sex == 'F'), "Weight"]
Male_Weight <- student[which(student$Sex == 'M'), "Weight"]

# Find the mean of both arrays 
means <- c(mean(Female_Weight), mean(Male_Weight))

# Syntax indicates Weight as a function of Sex 
boxplot(student$Weight ~ student$Sex, ylab= "Weight", xlab= "Sex")

# Plot means on boxplots in blue 
points(means, col= "blue")



# Subset data set to return only female weights, and then only male weights 
Female_Weight <- student[which(student$Sex == 'F'), "Weight"]
Male_Weight <- student[which(student$Sex == 'M'), "Weight"]

# Find the mean of both arrays 
means <- c(mean(Female_Weight), mean(Male_Weight))

# Syntax indicates Weight as a function of Sex 
boxplot(student$Weight ~ student$Sex, ylab= "Weight", xlab= "Sex")

# Plot means on boxplots in blue 
points(means, col= "blue")



# call the ggplot2 package
library(ggplot2)

student$Sex <- factor(student$Sex, levels = c("F","M"),
                      labels = c("Female", "Male"))
ggplot(data = student, aes(x = Sex, y = Weight, fill = Sex)) + 
  geom_boxplot() + stat_summary(fun.y = mean,
                                color = "black", geom = "point",
                                shape = 18, size = 3)



# call the ggplot2 package
library(ggplot2)

student$Sex <- factor(student$Sex, levels = c("F","M"),
                      labels = c("Female", "Male"))
ggplot(data = student, aes(x = Sex, y = Weight, fill = Sex)) + 
  geom_boxplot() + stat_summary(fun.y = mean,
                                color = "black", geom = "point",
                                shape = 18, size = 3)



# Notice here how you can create the BMI column in the data set just by 
# naming it 
student$BMI <- student$Weight / (student$Height)**2 * 703
head(student, n=5)



# Notice here how you can create the BMI column in the data set just by 
# naming it 
student$BMI <- student$Weight / (student$Height)**2 * 703
head(student, n=5)



# Notice the use of the ifelse() function for a single condition
student$BMI_Class <- ifelse(student$BMI<19.0, "Underweight", "Healthy")
head(student, n=5)



# Notice the use of the ifelse() function for a single condition
student$BMI_Class <- ifelse(student$BMI<19.0, "Underweight", "Healthy")
head(student, n=5)



student$LogWeight <- log(student$Weight)
student$ExpAge <- exp(student$Age)
student$SqrtHeight <- sqrt(student$Height)
student$BMI_Neg <- ifelse(student$BMI < 19.0, -student$BMI, student$BMI)
student$BMI_Pos <- abs(student$BMI_Neg)

# Create a Boolean variable
student$BMI_Check <- (student$BMI == student$BMI_Pos)
head(student, n=5)



student$LogWeight <- log(student$Weight)
student$ExpAge <- exp(student$Age)
student$SqrtHeight <- sqrt(student$Height)
student$BMI_Neg <- ifelse(student$BMI < 19.0, -student$BMI, student$BMI)
student$BMI_Pos <- abs(student$BMI_Neg)

# Create a Boolean variable
student$BMI_Check <- (student$BMI == student$BMI_Pos)
head(student, n=5)



# -c() function tells R not to select the columns listed
student <- subset(student, select = -c(LogWeight, ExpAge, SqrtHeight,
                                       BMI_Neg, BMI_Pos, BMI_Check))
head(student, n=5)



# -c() function tells R not to select the columns listed
student <- subset(student, select = -c(LogWeight, ExpAge, SqrtHeight,
                                       BMI_Neg, BMI_Pos, BMI_Check))
head(student, n=5)



student <- student[order(student$Age), ]
# Notice that R uses a stable sorting algorithm by default
head(student, n=5)



student <- student[order(student$Age), ]
# Notice that R uses a stable sorting algorithm by default
head(student, n=5)



student <- student[order(student$Sex), ]
# Notice that the data is now sorted first by Sex and then within Sex by Age 
head(student, n=5)



student <- student[order(student$Sex), ]
# Notice that the data is now sorted first by Sex and then within Sex by Age 
head(student, n=5)



# Notice the syntax of Age, Height, Weight, and BMI as a function of Sex 
aggregate(cbind(Age, Height, Weight, BMI) ~ Sex, student, mean)



# Notice the syntax of Age, Height, Weight, and BMI as a function of Sex 
aggregate(cbind(Age, Height, Weight, BMI) ~ Sex, student, mean)



# Look at the tail of the data currently
tail(student, n=5)



# Look at the tail of the data currently
tail(student, n=5)



# rbind.data.frame() function binds two data frames together by rows 
student <- rbind.data.frame(student, data.frame(Name='Jane', Sex = 'F',
                                                Age = 14, Height = 56.3,
                                                Weight = 77.0,
                                                BMI = 17.077695,
                                                BMI_Class = 'Underweight'))
tail(student, n=5)



# rbind.data.frame() function binds two data frames together by rows 
student <- rbind.data.frame(student, data.frame(Name='Jane', Sex = 'F',
                                                Age = 14, Height = 56.3,
                                                Weight = 77.0,
                                                BMI = 17.077695,
                                                BMI_Class = 'Underweight'))
tail(student, n=5)



toKG <- function(lb) {
  return(0.45359237 * lb)
}

student$Weight_KG <- toKG(student$Weight)
head(student, n=5)



toKG <- function(lb) {
  return(0.45359237 * lb)
}

student$Weight_KG <- toKG(student$Weight)
head(student, n=5)



# Notice the use of the fish data set because it has some missing 
# observations 
fish <- read.csv('/Users/fish.csv')

# First sort by Weight, requesting those with NA for Weight first 
fish <- fish[order(fish$Weight, na.last=FALSE), ]
head(fish, n=5)



# Notice the use of the fish data set because it has some missing 
# observations 
fish <- read.csv('/Users/fish.csv')

# First sort by Weight, requesting those with NA for Weight first 
fish <- fish[order(fish$Weight, na.last=FALSE), ]
head(fish, n=5)



new_fish <- na.omit(fish)
head(new_fish, n=5)



new_fish <- na.omit(fish)
head(new_fish, n=5)



# Notice the use of the student data set again, however we want to reload 
# it without the changes we've made previously  
student <- read.csv('/Users/class.csv')
student1 <- subset(student, select=c(Name, Sex, Age))
head(student1, n=5)



# Notice the use of the student data set again, however we want to reload 
# it without the changes we've made previously  
student <- read.csv('/Users/class.csv')
student1 <- subset(student, select=c(Name, Sex, Age))
head(student1, n=5)



student2 <- subset(student, select=c(Name, Height, Weight))
head(student2, n=5)



student2 <- subset(student, select=c(Name, Height, Weight))
head(student2, n=5)



new <- merge(student1, student2)
head(new, n=5)



new <- merge(student1, student2)
head(new, n=5)



all.equal(student, new)



all.equal(student, new)



newstudent1 <- subset(student, select=c(Name, Sex, Age))
head(newstudent1, n=5)



newstudent1 <- subset(student, select=c(Name, Sex, Age))
head(newstudent1, n=5)



newstudent2 <- subset(student, select=c(Height, Weight))
head(newstudent2, n=5)



newstudent2 <- subset(student, select=c(Height, Weight))
head(newstudent2, n=5)



new2 <- cbind(newstudent1, newstudent2)
head(new2, n=5)



new2 <- cbind(newstudent1, newstudent2)
head(new2, n=5)



all.equal(student, new2)



all.equal(student, new2)



# Notice we are using a new data set that needs to be read into the 
# environment
price <- read.csv('/Users/price.csv')

# call the dplyr package
library(dplyr)

# The following code is used to remove the "," and "$" characters from the 
# ACTUAL column so that values can be summed 
price$ACTUAL <- gsub('[$]', '', price$ACTUAL)
price$ACTUAL <- as.numeric(gsub(',', '', price$ACTUAL))

filtered = group_by(price, COUNTRY, STATE, PRODTYPE, PRODUCT)
basic_sum = summarise(filtered, REVENUE = sum(ACTUAL))
head(basic_sum, n=5)



# Notice we are using a new data set that needs to be read into the 
# environment
price <- read.csv('/Users/price.csv')

# call the dplyr package
library(dplyr)

# The following code is used to remove the "," and "$" characters from the 
# ACTUAL column so that values can be summed 
price$ACTUAL <- gsub('[$]', '', price$ACTUAL)
price$ACTUAL <- as.numeric(gsub(',', '', price$ACTUAL))

filtered = group_by(price, COUNTRY, STATE, PRODTYPE, PRODUCT)
basic_sum = summarise(filtered, REVENUE = sum(ACTUAL))
head(basic_sum, n=5)



print(unique(price$STATE))



print(unique(price$STATE))



# Notice we are using a new data set that needs to be read into the 
# environment
iris <- read.csv('/Users/iris.csv')
features <- subset(iris, select = -c(Target))

pca <- prcomp(x = features, scale = TRUE)
print(pca)



# Notice we are using a new data set that needs to be read into the 
# environment
iris <- read.csv('/Users/iris.csv')
features <- subset(iris, select = -c(Target))

pca <- prcomp(x = features, scale = TRUE)
print(pca)



# Set the sample size of the training data
smp_size <- floor(0.7 * nrow(iris))

# set.seed() is used to specify a seed for a random integer so that the 
# results are reproducible
set.seed(29)
train_ind <- sample(seq_len(nrow(iris)), size = smp_size)

train <- iris[train_ind, ]
test <- iris[-train_ind, ]

write.csv(train, file = "/Users/iris_train_R.csv")
write.csv(test, file = "/Users/iris_test_R.csv")



# Set the sample size of the training data
smp_size <- floor(0.7 * nrow(iris))

# set.seed() is used to specify a seed for a random integer so that the 
# results are reproducible
set.seed(29)
train_ind <- sample(seq_len(nrow(iris)), size = smp_size)

train <- iris[train_ind, ]
test <- iris[-train_ind, ]

write.csv(train, file = "/Users/iris_train_R.csv")
write.csv(test, file = "/Users/iris_test_R.csv")



# Notice we are using a new data set that needs to be read into the 
# environment
tips <- read.csv('/Users/tips.csv')

# The following code is used to determine if the individual left more 
# than a 15% tip 
tips$fifteen <- 0.15 * tips$total_bill
tips$greater15 <- ifelse(tips$tip > tips$fifteen, 1, 0)

# Notice the syntax of greater15 as a function of total_bill 
# You could fit the model of greater15 as a function of all
# other variables with "greater15 ~ ."
logreg <- glm(greater15 ~ total_bill, data = tips,
              family = "binomial"(link='logit'))
summary(logreg)



# Notice we are using a new data set that needs to be read into the 
# environment
tips <- read.csv('/Users/tips.csv')

# The following code is used to determine if the individual left more 
# than a 15% tip 
tips$fifteen <- 0.15 * tips$total_bill
tips$greater15 <- ifelse(tips$tip > tips$fifteen, 1, 0)

# Notice the syntax of greater15 as a function of total_bill 
# You could fit the model of greater15 as a function of all
# other variables with "greater15 ~ ."
logreg <- glm(greater15 ~ total_bill, data = tips,
              family = "binomial"(link='logit'))
summary(logreg)



# Notice the syntax of tip as function of total_bill
linreg <- lm(tip ~ total_bill, data = tips)
summary(linreg)



# Notice the syntax of tip as function of total_bill
linreg <- lm(tip ~ total_bill, data = tips)
summary(linreg)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/tips_train.csv')
test <- read.csv('/Users/tips_test.csv')

train$fifteen <- 0.15 * train$total_bill
train$greater15 <- ifelse(train$tip > train$fifteen, 1, 0)
test$fifteen <- 0.15 * test$total_bill
test$greater15 <- ifelse(test$tip > test$fifteen, 1, 0)

logreg <- glm(greater15 ~ total_bill, data = train,
              family = "binomial"(link='logit'))
summary(logreg)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/tips_train.csv')
test <- read.csv('/Users/tips_test.csv')

train$fifteen <- 0.15 * train$total_bill
train$greater15 <- ifelse(train$tip > train$fifteen, 1, 0)
test$fifteen <- 0.15 * test$total_bill
test$greater15 <- ifelse(test$tip > test$fifteen, 1, 0)

logreg <- glm(greater15 ~ total_bill, data = train,
              family = "binomial"(link='logit'))
summary(logreg)



# Prediction on testing data
predictions <- predict(logreg, test, type = 'response')
predY <- ifelse(predictions < 0.5, 0, 1)

# If the prediction probability is less than 0.5, classify this as a 0
# and otherwise classify as a 1.  This isn't the best method -- a better 
# method would be randomly assigning a 0 or 1 when a probability of 0.5 
# occurrs, but this insures that results are consistent 

# Determine how many were correctly classified
Results <- ifelse(predY == test$greater15, "Correct", "Wrong")
table(Results)



# Prediction on testing data
predictions <- predict(logreg, test, type = 'response')
predY <- ifelse(predictions < 0.5, 0, 1)

# If the prediction probability is less than 0.5, classify this as a 0
# and otherwise classify as a 1.  This isn't the best method -- a better 
# method would be randomly assigning a 0 or 1 when a probability of 0.5 
# occurrs, but this insures that results are consistent 

# Determine how many were correctly classified
Results <- ifelse(predY == test$greater15, "Correct", "Wrong")
table(Results)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# Fit a linear regression model
# The "." character tells the model to use all variables except the response 
# variabe (Target)
linreg <- lm(Target ~ ., data = train)
summary(linreg)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# Fit a linear regression model
# The "." character tells the model to use all variables except the response 
# variabe (Target)
linreg <- lm(Target ~ ., data = train)
summary(linreg)



# Predict on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(linreg, newdata = test)

# Compute the squared difference between predicted tip and actual tip 
prediction$sq_diff <- (prediction$predY - test$Target)**2

# Compute the mean of the squared differences (mean squared error) 
# as an assessment of the model 
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Predict on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(linreg, newdata = test)

# Compute the squared difference between predicted tip and actual tip 
prediction$sq_diff <- (prediction$predY - test$Target)**2

# Compute the mean of the squared differences (mean squared error) 
# as an assessment of the model 
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the tree package
library(tree)

treeMod <- tree(Target ~ ., data = train, method = "class")

# Plot the decision tree
plot(treeMod)
text(treeMod)



# Notice we are using new data sets that need to be read into the environment
train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the tree package
library(tree)

treeMod <- tree(Target ~ ., data = train, method = "class")

# Plot the decision tree
plot(treeMod)
text(treeMod)



# Determine variable importance
summary(treeMod)



# Determine variable importance
summary(treeMod)



# Prediction on testing data
out <- predict(treeMod, test)
out <- unname(out)
predY <- ifelse(out < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



# Prediction on testing data
out <- predict(treeMod, test)
out <- unname(out)
predY <- ifelse(out < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

treeMod <- tree(Target ~ ., data = train)

# Plot the decision tree
plot(treeMod)
text(treeMod)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

treeMod <- tree(Target ~ ., data = train)

# Plot the decision tree
plot(treeMod)
text(treeMod)



# Determine variable importance
summary(treeMod)



# Determine variable importance
summary(treeMod)



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(treeMod, newdata = test)

# Determine mean squared error
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(treeMod, newdata = test)

# Determine mean squared error
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the randomForest package
library(randomForest)
set.seed(29)

# as.factor() since classification model
rfMod <- randomForest(as.factor(Target) ~ ., data = train)

# Determine variable importance
var_import <- importance(rfMod)
var_import <- data.frame(sort(var_import, decreasing = TRUE,
                              index.return = TRUE))
var_import$MeanDecreaseGini <- var_import$x
var_import$X <- var_import$ix - 1
var_import <- subset(var_import, select = -c(ix, x))
head(var_import, n=5)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the randomForest package
library(randomForest)
set.seed(29)

# as.factor() since classification model
rfMod <- randomForest(as.factor(Target) ~ ., data = train)

# Determine variable importance
var_import <- importance(rfMod)
var_import <- data.frame(sort(var_import, decreasing = TRUE,
                              index.return = TRUE))
var_import$MeanDecreaseGini <- var_import$x
var_import$X <- var_import$ix - 1
var_import <- subset(var_import, select = -c(ix, x))
head(var_import, n=5)



# Prediction on testing data
predY <- predict(rfMod, test)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



# Prediction on testing data
predY <- predict(rfMod, test)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the randomForest package
library(randomForest)
set.seed(29)

rfMod <- randomForest(Target ~ ., data = train)

# Determine variable importance
var_import <- importance(rfMod)
var_import <- data.frame(sort(var_import, decreasing = TRUE,
                              index.return = TRUE))
var_import$MeanDecreaseGini <- var_import$x
var_import$X <- var_import$ix - 1
var_import <- subset(var_import, select = -c(ix, x))
head(var_import, n=5)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the randomForest package
library(randomForest)
set.seed(29)

rfMod <- randomForest(Target ~ ., data = train)

# Determine variable importance
var_import <- importance(rfMod)
var_import <- data.frame(sort(var_import, decreasing = TRUE,
                              index.return = TRUE))
var_import$MeanDecreaseGini <- var_import$x
var_import$X <- var_import$ix - 1
var_import <- subset(var_import, select = -c(ix, x))
head(var_import, n=5)



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(rfMod, newdata = test)

# Determine mean squared error
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY = predict(rfMod, newdata = test)

# Determine mean squared error
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the gbm package
library(gbm)
set.seed(29)

# distribution = "bernoulli" is appropriate when there are only 2 
# unique values
# n.trees = total number of trees to fit which is analogous to the number 
# of iterations
# shrinkage = learning rate or step-size reduction, whereas a lower 
# learning rate requires more iterations
gbMod <- gbm(Target ~ ., distribution = "bernoulli", data = train,
             n.trees = 2500, shrinkage = .01)

# Determine variable importance
var_import <- summary(gbMod)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the gbm package
library(gbm)
set.seed(29)

# distribution = "bernoulli" is appropriate when there are only 2 
# unique values
# n.trees = total number of trees to fit which is analogous to the number 
# of iterations
# shrinkage = learning rate or step-size reduction, whereas a lower 
# learning rate requires more iterations
gbMod <- gbm(Target ~ ., distribution = "bernoulli", data = train,
             n.trees = 2500, shrinkage = .01)

# Determine variable importance
var_import <- summary(gbMod)



head(var_import, n=5)



head(var_import, n=5)



# Prediction on testing data
out <- predict(object = gbMod, newdata = test,
                         type = "response", n.trees = 2500)
predY <- ifelse(out < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



# Prediction on testing data
out <- predict(object = gbMod, newdata = test,
                         type = "response", n.trees = 2500)
predY <- ifelse(out < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the gbm package
library(gbm)
set.seed(29)

gbMod <- gbm(Target ~ ., data = train, distribution = "gaussian",
             n.trees = 2500, shrinkage = .01)

# Determine variable importance
var_import <- summary(gbMod)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the gbm package
library(gbm)
set.seed(29)

gbMod <- gbm(Target ~ ., data = train, distribution = "gaussian",
             n.trees = 2500, shrinkage = .01)

# Determine variable importance
var_import <- summary(gbMod)



head(var_import, n=5)



head(var_import, n=5)



# Predict the Target in the testing data, remembeing to multiply by 50
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- predict(object = gbMod, newdata = test,
                            type = "response", n.trees = 2500)

# Compute mean squared error 
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Predict the Target in the testing data, remembeing to multiply by 50
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- predict(object = gbMod, newdata = test,
                            type = "response", n.trees = 2500)

# Compute mean squared error 
prediction$sq_diff <- (prediction$predY - test$Target)**2
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the xgboost package
library(xgboost)
set.seed(29)

# Fit the model
xgbMod <- xgboost(data.matrix(subset(train, select = -c(Target))),
                 data.matrix(train$Target), max_depth = 3, nrounds = 2,
                 objective = "binary:logistic", n_estimators = 2500,
                 shrinkage = .01)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the xgboost package
library(xgboost)
set.seed(29)

# Fit the model
xgbMod <- xgboost(data.matrix(subset(train, select = -c(Target))),
                 data.matrix(train$Target), max_depth = 3, nrounds = 2,
                 objective = "binary:logistic", n_estimators = 2500,
                 shrinkage = .01)



# Prediction on testing data
predictions <- predict(xgbMod,
                       data.matrix(subset(test,
                                          select = -c(Target))))
predY <- ifelse(predictions < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



# Prediction on testing data
predictions <- predict(xgbMod,
                       data.matrix(subset(test,
                                          select = -c(Target))))
predY <- ifelse(predictions < 0.5, 0, 1)

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the xgboost package
library(xgboost)
set.seed(29)

# Fit the model
xgbMod <- xgboost(data.matrix(subset(train, select = -c(Target))),
                 data.matrix(train$Target), max_depth = 3, nrounds = 10,
                 n_estimators = 2500, shrinkage = .01)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the xgboost package
library(xgboost)
set.seed(29)

# Fit the model
xgbMod <- xgboost(data.matrix(subset(train, select = -c(Target))),
                 data.matrix(train$Target), max_depth = 3, nrounds = 10,
                 n_estimators = 2500, shrinkage = .01)



# Prediction on testing
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- predict(xgbMod,
                    data.matrix(subset(test, select = -c(Target))))

# Compute the squared difference between predicted tip and actual tip 
prediction$sq_diff <- (prediction$predY - test$Target)**2

# Compute the mean of the squared differences (mean squared error) 
# as an assessment of the model 
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



# Prediction on testing
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- predict(xgbMod,
                    data.matrix(subset(test, select = -c(Target))))

# Compute the squared difference between predicted tip and actual tip 
prediction$sq_diff <- (prediction$predY - test$Target)**2

# Compute the mean of the squared differences (mean squared error) 
# as an assessment of the model 
mean_sq_error <- mean(prediction$sq_diff)
print(mean_sq_error)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the e1071 package
library(e1071)

# Fit a support vector classification model 
svMod <- svm(Target ~ ., train, type = 'C-classification', kernel = 'linear', scale = FALSE)



train <- read.csv('/Users/breastcancer_train.csv')
test <- read.csv('/Users/breastcancer_test.csv')

# call the e1071 package
library(e1071)

# Fit a support vector classification model 
svMod <- svm(Target ~ ., train, type = 'C-classification', kernel = 'linear', scale = FALSE)



# Prediction on testing data
predY <- unname(predict(svMod, subset(test, select = -c(Target))))

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



# Prediction on testing data
predY <- unname(predict(svMod, subset(test, select = -c(Target))))

# Determine how many were correctly classified
Results <- ifelse(test$Target == predY, "Correct", "Wrong")
table(Results)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the e1071 package
library(e1071)

svMod <- svm(Target ~ ., train, scale = FALSE)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the e1071 package
library(e1071)

svMod <- svm(Target ~ ., train, scale = FALSE)



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- unname(predict(svMod, test))
prediction$sq_diff <- (prediction$predY - test$Target)**2
print(mean(prediction$sq_diff))



# Prediction on testing data
prediction = data.frame(matrix(ncol = 0, nrow = nrow(test)))
prediction$predY <- unname(predict(svMod, test))
prediction$sq_diff <- (prediction$predY - test$Target)**2
print(mean(prediction$sq_diff))



# Notice we are using new data sets
train <- read.csv('/Users/digits_train.csv')
test <- read.csv('/Users/digits_test.csv')

trainInputs <- subset(train, select = -c(Target))
testInputs <- subset(test, select = -c(Target))

# call the RSNNS package
library(RSNNS)
set.seed(29)

trainTarget <- decodeClassLabels(train$Target)
testTarget <- decodeClassLabels(test$Target)

# Fit neural network regression model
nnMod <- mlp(trainInputs, trainTarget, size = c(100), maxit = 200)



# Notice we are using new data sets
train <- read.csv('/Users/digits_train.csv')
test <- read.csv('/Users/digits_test.csv')

trainInputs <- subset(train, select = -c(Target))
testInputs <- subset(test, select = -c(Target))

# call the RSNNS package
library(RSNNS)
set.seed(29)

trainTarget <- decodeClassLabels(train$Target)
testTarget <- decodeClassLabels(test$Target)

# Fit neural network regression model
nnMod <- mlp(trainInputs, trainTarget, size = c(100), maxit = 200)



# Prediction on testing data 
predictions <- predict(nnMod, testInputs)

# Determine how many were correctly classified
confusionMatrix(testTarget, predictions)



# Prediction on testing data 
predictions <- predict(nnMod, testInputs)

# Determine how many were correctly classified
confusionMatrix(testTarget, predictions)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the RSNNS package
library(RSNNS)
set.seed(29)

# Scale input data
scaled_train <- data.frame(scale(subset(train, select = -c(Target))))
scaled_test <- data.frame(scale(subset(test, select = -c(Target))))

# Fit neural network regression model, dividing target by 50 for scaling
nnMod <- mlp(scaled_train, train$Target / 50, maxit = 250, size = c(100))



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

# call the RSNNS package
library(RSNNS)
set.seed(29)

# Scale input data
scaled_train <- data.frame(scale(subset(train, select = -c(Target))))
scaled_test <- data.frame(scale(subset(test, select = -c(Target))))

# Fit neural network regression model, dividing target by 50 for scaling
nnMod <- mlp(scaled_train, train$Target / 50, maxit = 250, size = c(100))



# Assess against testing data, remembering to multiply by 50
preds = data.frame(matrix(ncol = 0, nrow = nrow(test)))
preds$predY <- predict(nnMod, scaled_test)*50
preds$sq_error <- (preds$predY - test$Target)**2
print(mean(preds$sq_error))



# Assess against testing data, remembering to multiply by 50
preds = data.frame(matrix(ncol = 0, nrow = nrow(test)))
preds$predY <- predict(nnMod, scaled_test)*50
preds$sq_error <- (preds$predY - test$Target)**2
print(mean(preds$sq_error))



iris = read.csv('/Users/iris.csv')

iris$Species = ifelse(iris$Target == 0, "Setosa",
                      ifelse(iris$Target == 1, "Versicolor", "Virginica"))

features <- as.matrix(subset(iris, select = c(PetalLength, PetalWidth,
                                              SepalLength, SepalWidth)))

set.seed(29)

kmeans <- kmeans(features, 3)

table(iris$Species, kmeans$cluster)



iris = read.csv('/Users/iris.csv')

iris$Species = ifelse(iris$Target == 0, "Setosa",
                      ifelse(iris$Target == 1, "Versicolor", "Virginica"))

features <- as.matrix(subset(iris, select = c(PetalLength, PetalWidth,
                                              SepalLength, SepalWidth)))

set.seed(29)

kmeans <- kmeans(features, 3)

table(iris$Species, kmeans$cluster)



# call the kernlab package
library(kernlab)

set.seed(29)

spectral <- specc(features, centers = 3, iterations = 10, nystrom.red = TRUE)

labels <- as.data.frame(spectral)

table(iris$Species, labels$spectral)



# call the kernlab package
library(kernlab)

set.seed(29)

spectral <- specc(features, centers = 3, iterations = 10, nystrom.red = TRUE)

labels <- as.data.frame(spectral)

table(iris$Species, labels$spectral)



set.seed(29)

hclust <- hclust(dist(features), method = "ward.D2")

table(iris$Species, cutree(hclust, 3))



set.seed(29)

hclust <- hclust(dist(features), method = "ward.D2")

table(iris$Species, cutree(hclust, 3))



# call the dbscan package
library(dbscan)

set.seed(29)

# eps = 0.5 is default in Python
dbscan <- dbscan(features, eps = 0.5)

table(iris$Species, dbscan$cluster)



# call the dbscan package
library(dbscan)

set.seed(29)

# eps = 0.5 is default in Python
dbscan <- dbscan(features, eps = 0.5)

table(iris$Species, dbscan$cluster)



# call the kohonen package
library(kohonen)

# Seed chosen to match SAS and R results
set.seed(5)

fit <- som(features, mode = "online", somgrid(4, 4, "rectangular"))

plot(fit, type = "dist.neighbour", shape = "straight")



# call the kohonen package
library(kohonen)

# Seed chosen to match SAS and R results
set.seed(5)

fit <- som(features, mode = "online", somgrid(4, 4, "rectangular"))

plot(fit, type = "dist.neighbour", shape = "straight")



# Read in new data set
air <- read.csv('/Users/air.csv')

air_series <- air$AIR

plot.ts(air_series, ylab="Air")



# Read in new data set
air <- read.csv('/Users/air.csv')

air_series <- air$AIR

plot.ts(air_series, ylab="Air")



a_fit <- arima(air_series, order = c(0,1,1),
               seasonal = list(order = c(0,1,1), period = 12),
               method = "ML")

# call the forecast package
library(forecast)

a_forecast <- forecast(a_fit, 24)

plot(a_forecast, xlab = "Month", ylab = "Air")



a_fit <- arima(air_series, order = c(0,1,1),
               seasonal = list(order = c(0,1,1), period = 12),
               method = "ML")

# call the forecast package
library(forecast)

a_forecast <- forecast(a_fit, 24)

plot(a_forecast, xlab = "Month", ylab = "Air")



# Read in new data set
usecon <- read.csv('/Users/usecon.csv')

petrol_series <- usecon$PETROL

petrol <- ts(petrol_series, frequency = 12)

plot.ts(petrol, ylab="Petrol")



# Read in new data set
usecon <- read.csv('/Users/usecon.csv')

petrol_series <- usecon$PETROL

petrol <- ts(petrol_series, frequency = 12)

plot.ts(petrol, ylab="Petrol")



# call the forecast package
library(forecast)

ses_fit <- ses(petrol, h=24, alpha = 0.9999)

plot(ses_fit, xlab = "Month", ylab = "Petrol")



# call the forecast package
library(forecast)

ses_fit <- ses(petrol, h=24, alpha = 0.9999)

plot(ses_fit, xlab = "Month", ylab = "Petrol")



vehicle_series <- usecon$VEHICLES

vehicle <- ts(vehicle_series, frequency = 12)

plot.ts(vehicle, ylab="Vehicle")



vehicle_series <- usecon$VEHICLES

vehicle <- ts(vehicle_series, frequency = 12)

plot.ts(vehicle, ylab="Vehicle")



# call the forecast package
library(forecast)

add_fit <- HoltWinters(vehicle, seasonal = "additive")

add_forecast <- forecast(add_fit, 24)

plot(add_forecast)



# call the forecast package
library(forecast)

add_fit <- HoltWinters(vehicle, seasonal = "additive")

add_forecast <- forecast(add_fit, 24)

plot(add_forecast)



air <- read.csv('/Users/air.csv')

# call the prophet & dplyr packages
library(prophet)
library(dplyr)

air_df <- data.frame(matrix(ncol = 0, nrow = nrow(air)))

air_df$ds <- as.Date(air$DATE, format = "%m/%d/%Y")
air_df$y <- air$AIR

m <- prophet(air_df, yearly.seasonality = TRUE, weekly.seasonality = FALSE)



air <- read.csv('/Users/air.csv')

# call the prophet & dplyr packages
library(prophet)
library(dplyr)

air_df <- data.frame(matrix(ncol = 0, nrow = nrow(air)))

air_df$ds <- as.Date(air$DATE, format = "%m/%d/%Y")
air_df$y <- air$AIR

m <- prophet(air_df, yearly.seasonality = TRUE, weekly.seasonality = FALSE)



future <- make_future_dataframe(m, periods = 24, freq = "month")

forecast <- predict(m, future)

plot(m, forecast)



future <- make_future_dataframe(m, periods = 24, freq = "month")

forecast <- predict(m, future)

plot(m, forecast)



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

set.seed(29)

# Random Forest Regression Model
# call the randomForest package
library(randomForest)

rfMod <- randomForest(Target ~ ., data = train)

# Evaluation on training data
predY <- predict(rfMod, train)
predY <- unname(predY)

# Determine coefficient of determination score
r2_rf <- 1 - ( (sum((train$Target - 
                       predY)**2)) / (sum((train$Target - 
                                           mean(train$Target))**2)) )
print(paste0("Random forest regression model r^2 score (coefficient of determination): ", r2_rf))



train <- read.csv('/Users/boston_train.csv')
test <- read.csv('/Users/boston_test.csv')

set.seed(29)

# Random Forest Regression Model
# call the randomForest package
library(randomForest)

rfMod <- randomForest(Target ~ ., data = train)

# Evaluation on training data
predY <- predict(rfMod, train)
predY <- unname(predY)

# Determine coefficient of determination score
r2_rf <- 1 - ( (sum((train$Target - 
                       predY)**2)) / (sum((train$Target - 
                                           mean(train$Target))**2)) )
print(paste0("Random forest regression model r^2 score (coefficient of determination): ", r2_rf))



# Random Forest Regression Model (rfMod) 

# Evaluation on testing data
predY <- predict(rfMod, test)
predY <- unname(predY)

# Determine coefficient of determination score
r2_rf = 1 - ( (sum((test$Target - 
                      predY)**2)) / (sum((test$Target - 
                                          mean(test$Target))**2)) )
print(paste0("Random forest regression model r^2 score (coefficient of determination): ", r2_rf))



# Random Forest Regression Model (rfMod) 

# Evaluation on testing data
predY <- predict(rfMod, test)
predY <- unname(predY)

# Determine coefficient of determination score
r2_rf = 1 - ( (sum((test$Target - 
                      predY)**2)) / (sum((test$Target - 
                                          mean(test$Target))**2)) )
print(paste0("Random forest regression model r^2 score (coefficient of determination): ", r2_rf))



train <- read.csv('/Users/digits_train.csv')
test <- read.csv('/Users/digits_test.csv')

set.seed(29)

# Random Forest Classification Model
# call the randomForest package
library(randomForest)

rfMod <- randomForest(as.factor(Target) ~ ., data = train)

# Evaluation on training data
predY <- predict(rfMod, train)
predY <- unname(predY)

# Determine accuracy score
accuracy_rf <- (1/nrow(train)) * sum(as.numeric(predY == train$Target))
print(paste0("Random forest model accuracy: ", accuracy_rf))



train <- read.csv('/Users/digits_train.csv')
test <- read.csv('/Users/digits_test.csv')

set.seed(29)

# Random Forest Classification Model
# call the randomForest package
library(randomForest)

rfMod <- randomForest(as.factor(Target) ~ ., data = train)

# Evaluation on training data
predY <- predict(rfMod, train)
predY <- unname(predY)

# Determine accuracy score
accuracy_rf <- (1/nrow(train)) * sum(as.numeric(predY == train$Target))
print(paste0("Random forest model accuracy: ", accuracy_rf))



# Random Forest Classification Model (rfMod)

# Evaluation on testing data
predY <- predict(rfMod, test)
predY <- unname(predY)

# Determine accuracy score
accuracy_rf <- (1/nrow(test)) * sum(as.numeric(predY == test$Target))
print(paste0("Random forest model accuracy: ", accuracy_rf))



# Random Forest Classification Model (rfMod)

# Evaluation on testing data
predY <- predict(rfMod, test)
predY <- unname(predY)

# Determine accuracy score
accuracy_rf <- (1/nrow(test)) * sum(as.numeric(predY == test$Target))
print(paste0("Random forest model accuracy: ", accuracy_rf))



my_array <- c(1, 3, 5, 9)
print(my_array)



my_array <- c(1, 3, 5, 9)
print(my_array)



print(my_array[1])



print(my_array[1])



student <- read.csv('/Users/class.csv')
values <- student$Age
names(values) <- student$Name
print(values["James"])



student <- read.csv('/Users/class.csv')
values <- student$Age
names(values) <- student$Name
print(values["James"])



list1 <- list('item1', 102)
print(list1)



list1 <- list('item1', 102)
print(list1)



print(list1[1])



print(list1[1])



