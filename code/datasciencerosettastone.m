
% This is a single line comment.
%{ This is a paragraph
comment %}




formatSpec = '%C%C%d%f%f';
student = readtable('class.csv', 'Delimiter', ',', 'Format', formatSpec);




student_xlsx = readtable('class.xlsx');




student_json = jsondecode(fileread('class.json'));




disp(size(student));




summary(student);




disp(student(1:5,:));





age = table2array(student(:,3));
disp(mean(age));




height = table2array(student(:,4));
disp(mean(height));




weight = table2array(student(:,5));
disp(mean(weight));




numeric_vars = student(:,{'Age', 'Height', 'Weight'});
statarray = grpstats(numeric_vars, [], {'min', 'median', 'mean', 'max'});
disp(statarray);




weight = table2array(student(:,5));
disp(std(weight));




disp(sum(weight));




disp(length(weight));




disp(max(weight));




disp(min(weight));




disp(median(weight));




tabulate(age);




sex = table2array(student(:,{'Sex'}));
tabulate(sex);




crosstable = varfun(@(x) length(x), student, 'GroupingVariables', {'Age' 'Sex'}, 'InputVariables', {});
disp(crosstable);




% Find the indices of those students who are females, and then get those observations
% from the student data frame.
females = student(student.Sex == 'F',:);
disp(females(1:5,:));




height_weight = cat(2,table2array(student(:,4)),table2array(student(:,5)));
disp(corr(height_weight));




Weight = table2array(student(:,{'Weight'}));
histogram(Weight, 40:20:160)
xlabel('Weight');
ylabel('Frequency');




boxplot(Weight);
mx = mean(Weight);
ylabel('Weight');
hold on
plot(mx, 'd')
hold off




Height = table2array(student(:,{'Height'}));
scatter(Height, Weight)
xticks(50:5:75)
yticks(40:20:160)
xlabel('Height')
ylabel('Weight')




scatter(Height, Weight)
xticks(50:5:75)
yticks(40:20:160)
xlabel('Height')
ylabel('Weight')
b = polyfit(Height, Weight,1);
m = b(1);
y_int = b(2);
lsline
annotation('textbox', [.2 .5 .3 .3], 'String', sprintf('Line: y = %fx + %f', m, y_int), ...
    'FitBoxToText', 'on');




Sex = table2array(student(:,{'Sex'}));
histogram(Sex)
xlabel('Sex')
ylabel('Frequency')




females = student(student.Sex == 'F',:);
males = student(student.Sex == 'M',:);
Female_Weight = table2array(females(:,{'Weight'}));
Male_Weight = table2array(males(:,{'Weight'}));
clf
boxplot(Weight, Sex);
means = [mean(Female_Weight), mean(Male_Weight)];
xlabel('Sex');
ylabel('Weight');
hold on
plot(means, 'd')
hold off




% The "./" (and similarly, ".^2") tells MATLAB to divide (and similarly, exponentiate)
% element-wise, instead of matrix-wise. 
student.BMI = student.Weight ./ student.Height .^ 2 * 703;
disp(student(1:5,:));




student.BMI_Class = student.Name;
for i = 1:size(student,1)
    if student.BMI(i) < 19.0
        student.BMI_Class(i) = 'Underweight';
    else
        student.BMI_Class(i) = 'Healthy';
    end
end
disp(student(1:5,:));




student.LogWeight = log(student.Weight);
student.ExpAge = exp(double(student.Age));
student.SqrtHeight = sqrt(student.Height);
student.BMI_Neg = student.BMI;
for i = 1:size(student,1)
    if student.BMI(i) < 19.0
        student.BMI_Neg(i) = -student.BMI(i);
    end
end
student.BMI_Pos = abs(student.BMI_Neg);
student.BMI_Check = (student.BMI_Pos == student.BMI);
disp(student(1:5,:));




student.LogWeight = [];
student.ExpAge = [];
student.SqrtHeight = [];
student.BMI_Neg = [];
student.BMI_Pos = [];
student.BMI_Check = [];
disp(student(1:5,:));




student = sortrows(student, 'Age');
disp(student(1:5,:));




student = sortrows(student, 'Sex');
disp(student(1:5,:));




group_means = grpstats(student, 'Sex', 'mean', 'DataVars', {'Age', 'Height', 'Weight', 'BMI'});
disp(group_means);




disp(student(15:19,:));




newObs = {'Name', 'Sex', 'Age', 'Height', 'Weight', 'BMI', 'BMI_Class';
    'Jane', 'F', 14, 56.3, 77.0, 17.077695, 'Underweight'};
newTable = dataset2table(cell2dataset(newObs));
student = vertcat(student,newTable);
disp(student(16:20,:));





% To create a user-defined function, create a new file in MATLAB with the function definition, 
% and save the file as the function_name.m.  Here, toKG.m would be:
% 
% function KG = toKG(lb);
%          KG = 0.45359237 * lb;
% end
student.Weight_KG = toKG(student.Weight);
disp(student(1:5,:));




formatSpec = '%C%f%f%f%f%f%f';
fish = readtable('fish.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
fish = sortrows(fish, 'Weight', 'descend');
disp(fish(1:5,:));




fish = rmmissing(fish);
disp(fish(1:5,:));




formatSpec = '%C%C%d%f%f';
student = readtable('class.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
student1 = student(:, {'Name', 'Sex', 'Age'});
disp(student1(1:5,:));




student2 = student(:, {'Name', 'Height', 'Weight'});
disp(student2(1:5,:));




new = join(student1, student2);
disp(new(1:5,:));




disp(isequal(student, new));




newstudent1 = student(:, {'Name', 'Sex', 'Age'});
disp(newstudent1(1:5,:));




newstudent2 = student(:, {'Height', 'Weight'});
disp(newstudent2(1:5,:));




new2 = [newstudent1, newstudent2];
disp(new2(1:5,:));




disp(isequal(student, new2));




% Currently there is not a MATLAB function for creating pivot tables, but only user-defined functions
% that could be used to create pivot tables.




price = readtable('price.xlsx');
disp(unique(price.STATE));




formatSpec = '%f%f%f%f%d';
iris = readtable('iris.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
features = table2array(iris(:, {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'}));
% Z-score function to scale
Zsc = @(x) (x-mean(x))./std(x);
features_scaled = Zsc(features);
disp(pca(features_scaled));




sizeIris = size(iris);
numRows = sizeIris(1);
% Set the seed of the random number generator
% for reproducibility.
rng(29);
[trainInd, valInd, testInd] = dividerand(numRows, 0.7, 0, 0.3);
train = iris(trainInd,:);
test = iris(testInd,:);
csvwrite('iris_train_ML.csv', table2array(train));
csvwrite('iris_test_ML.csv', table2array(test));




formatSpec = '%d%f%f%C%C%C%C%d';
tips = readtable('tips.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
tips.fifteen = 0.15 * tips.total_bill;
tips.greater15 = (tips.tip > tips.fifteen);
[b, dev, stats] = glmfit(tips.total_bill, tips.greater15, 'binomial', 'link', 'logit');
fprintf('The coefficients of the model are: %.3f and %.3f\n', b(1), b(2));
fprintf('The deviance of the fit of the fit is: %.3f\n', dev);
fprintf('Other statistics of the model are:\n');
disp(stats);




linreg = fitlm(tips,'tip~total_bill');
disp(linreg);




formatSpec = '%f%d%d%d%f';
train = readtable('tips_train.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
test = readtable('tips_test.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
train.fifteen = 0.15 * train.total_bill;
train.greater15 = (train.tip > train.fifteen);
test.fifteen = 0.15 * test.total_bill;
test.greater15 = (test.tip > test.fifteen);
[b, dev, stats] = glmfit(train.total_bill, train.greater15, 'binomial', 'link', 'logit');
fprintf('The coefficients of the model are: %.3f and %.3f\n', b(1), b(2));
fprintf('The deviance of the fit of the fit is: %.3f\n', dev);
fprintf('Other statistics of the model are:\n');
disp(stats);




predictions = glmval(b, test.total_bill, 'logit');
predY = round(predictions);
Results = strings(size(test,1),1);
for i = 1:size(test,1)
    if (predY(i) == test.greater15(i))
        Results(i) = 'Correct';
    else
        Results(i) = 'Wrong';
    end
end
tabulate(Results);




train = readtable('boston_train.xlsx');
test = readtable('boston_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
linreg = fitlm(table2array(x_train), y_train);
disp(linreg);




predictions = predict(linreg, table2array(x_test));
sq_diff = (predictions - y_test) .^ 2;
disp(mean(sq_diff));




train = readtable('breastcancer_train.xlsx');
test = readtable('breastcancer_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
treeMod = fitctree(x_train, y_train);
var_import = predictorImportance(treeMod);
var_import = var_import';
var_import(:,2) = var_import;
for i = 1:size(var_import,1)
    var_import(i,1) = i;
end
var_import = sortrows(var_import, 2, 'descend');
disp(var_import(1:5,:));




predictions = predict(treeMod, x_test);
Results = strings(size(test,1),1);
for i = 1:size(test,1)
    if (predictions(i) == y_test(i))
        Results(i) = 'Correct';
    else
        Results(i) = 'Wrong';
    end
end
tabulate(Results);




train = readtable('boston_train.xlsx');
test = readtable('boston_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
treeMod = fitrtree(x_train, y_train);
var_import = predictorImportance(treeMod);
var_import = var_import';
var_import(:,2) = var_import;
for i = 1:size(var_import,1)
    var_import(i,1) = i;
end
var_import = sortrows(var_import, 2, 'descend');
disp(var_import(1:5,:));




predictions = predict(treeMod, x_test);
sq_diff = (predictions - y_test) .^ 2;
disp(mean(sq_diff));




train = readtable('breastcancer_train.xlsx');
test = readtable('breastcancer_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
rfMod = fitrensemble(table2array(x_train), y_train, 'Method', 'bag');
var_import = predictorImportance(rfMod);
var_import = var_import';
var_import(:,2) = var_import;
for i = 1:size(var_import,1)
    var_import(i,1) = i;
end
var_import = sortrows(var_import, 2, 'descend');
disp(var_import(1:5,:));




predictions = predict(rfMod, table2array(x_test));
predictions = round(predictions);
Results = strings(size(test,1),1);
for i = 1:size(test,1)
    if (predictions(i) == y_test(i))
        Results(i) = 'Correct';
    else
        Results(i) = 'Wrong';
    end
end
tabulate(Results);




train = readtable('boston_train.xlsx');
test = readtable('boston_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
rfMod = fitrensemble(table2array(x_train), y_train, 'Method', 'bag');
var_import = predictorImportance(rfMod);
var_import = var_import';
var_import(:,2) = var_import;
for i = 1:size(var_import,1)
    var_import(i,1) = i;
end
var_import = sortrows(var_import, 2, 'descend');
disp(var_import(1:5,:));




predictions = predict(rfMod, table2array(x_test));
sq_diff = (predictions - y_test) .^ 2;
disp(mean(sq_diff));




train = readtable('breastcancer_train.xlsx');
test = readtable('breastcancer_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
svMod = fitcsvm(x_train, y_train);




predictions = predict(svMod, x_test);
Results = strings(size(test,1),1);
for i = 1:size(test,1)
    if (predictions(i) == y_test(i))
        Results(i) = 'Correct';
    else
        Results(i) = 'Wrong';
    end
end
tabulate(Results);




train = readtable('boston_train.xlsx');
test = readtable('boston_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
svMod = fitrsvm(x_train, y_train);




predictions = predict(svMod, x_test);
sq_diff = (predictions - y_test) .^ 2;
disp(mean(sq_diff));




formatSpec = '%f%f%f%f%d';
iris = readtable('iris.csv', 'Delimiter', ',', ...
    'Format', formatSpec);
features = table2array(iris(:, {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'}));
iris.Labels = strings(size(iris,1),1);
for i = 1:size(iris,1)
    if (iris.Target(i) == 0)
        iris.Labels(i) = 'Setosa';
    else
        if (iris.Target(i) == 1)
            iris.Labels(i) = 'Versicolor';
        else
            iris.Labels(i) = 'Virginica';
        end
    end
end
rng(29);
[labels, C] = kmeans(features, 3);
iris.Predictions = labels;
disp(crosstab(iris.Labels, iris.Predictions));




rng(29);
tree = linkage(features, 'ward', 'euclidean', 'savememory', 'on');
labels = cluster(tree, 'maxclust', 3);
iris.Predictions = labels;
disp(crosstab(iris.Labels, iris.Predictions));




air = readtable('air.xlsx');
plot(air.DATE, air.AIR);




model = arima('Constant',0,'D',1,'Seasonality',12,...
    'MALags',1,'SMALags',12);
est_model = estimate(model, air.AIR);
[yF, yMSE] = forecast(est_model, 24, 'Y0', air.AIR);
plot(yF);




train = readtable('boston_train.xlsx');
test = readtable('boston_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
rfMod = fitrensemble(table2array(x_train), y_train, 'Method', 'bag');
predictions = predict(rfMod, table2array(x_train));
r2_rf = 1 - ( (sum((y_train - predictions) .^ 2)) / (sum((y_train - mean(y_train)) .^ 2)) );
fprintf('Random forest regression model r^2 score (coefficient of determination): %.3f\n', r2_rf);




predictions = predict(rfMod, table2array(x_test));
r2_rf = 1 - ( (sum((y_test - predictions) .^ 2)) / (sum((y_test - mean(y_test)) .^ 2)) );
fprintf('Random forest regression model r^2 score (coefficient of determination): %.3f\n', r2_rf);




train = readtable('digits_train.xlsx');
test = readtable('digits_test.xlsx');
x_train = train;
x_train.Target = [];
y_train = train.Target;
x_test = test;
x_test.Target = [];
y_test = test.Target;
rng(29);
rfMod = fitrensemble(table2array(x_train), y_train, 'Method', 'bag');
predY = predict(rfMod, table2array(x_train));
predY = round(predY);
Results = zeros(size(train,1),1);
for i = 1:size(Results,1)
    if (predY(i) == y_train(i))
        Results(i) = 1;
    else
        Results(i) = 0;
    end
end
accuracy_rf = (1/size(x_train,1)) * sum(Results);
fprintf('Random forest model accuracy:  %.3f\n', accuracy_rf);




predY = predict(rfMod, table2array(x_test));
predY = round(predY);
Results = zeros(size(test,1),1);
for i = 1:size(Results,1)
    if (predY(i) == y_test(i))
        Results(i) = 1;
    else
        Results(i) = 0;
    end
end
accuracy_rf = (1/size(x_test,1)) * sum(Results);
fprintf('Random forest model accuracy:  %.3f\n', accuracy_rf);




my_matrix = [1 2 3; 4 5 6; 7 8 9]
disp(my_matrix(2,2));




my_vector = [1 3 5 9]
disp(my_vector(1));




