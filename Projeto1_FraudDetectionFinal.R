##################################################################################
#                                                                                #
# This project was developed during my studies in the course Big Data Analytics  #
# with R and Azure2.0, offered by Data Science Academy.                          #
# (www.datascienceacademy.com.br).                                               #
#                                                                                #
##################################################################################
#                                                                                #
# This project provides a simple predictive model to support the fraud detection #
# in traffic of clicks on mobile applications advertising. Three algorithms,     #
# including Random Forest, Naive Bayes and Support Vector Machines (SVM) were    #
# tested in this project.                                                        #
#                                                                                #
# It is relevant to mention that the train dataset was not used during the       #
# development of the model because the data set contains too much data to be     #
# processed in my machine, especially using R for this purpose. Therefore, the   #
# model creation was based on the train_sample data set.                         #
#                                                                                #
##################################################################################

##################################################################################
#                                                                                #
#                                   PART I                                       #
#                                                                                #
##################################################################################


############################### LOADING PACKAGES #################################

# Loading packages
library(data.table)
library(dplyr)
library(lubridate)
library(timetk)
library(caret)

library(corrplot)
library(ggplot2)
library(ROSE)
library(randomForest)
library(e1071)

# Set seed for reproducibility purposes
set.seed(12345)

##################### DATA LOADING AND PRE-PROCESSING ##########################

# Loading training_sample dataset
data = fread('data/train_sample.csv')

# Visualizing the data
head(data)

# Checking data types of variables
glimpse(data)

# Checking for na values, considering each variable
sapply(colnames(data), function(x) {sum(is.na(data[,..x]))})

# As the variable attributed_time presents more than 99% of missing data, this
# variable was removed.

# Transforming categorical variables to categorical type
data = data %>%
  select(-attributed_time) %>%
  mutate(ip = as.factor(ip),
         device = as.factor(device),
         os = as.factor(os),
         app = as.factor(app),
         channel = as.factor(channel),
         is_attributed = as.factor(is_attributed))

###################### PRELIMINAR EXPLORATORY ANALYSIS #########################

# Visualizing some descriptive statistics 
summary(data)

# Checking the number of categories, considering each variable
apply(X = data, MARGIN = 2, FUN = function(x) {length(unique(x))})

# From the above analysis, both ip and click_time presents a lot of categories,
# and therefore, shouldn't be used in the model prediction. In this case, the
# click_time variable is almost like an ID.
# Let's plot some graphs to visualize the variables distribution.

# Distribution of device
ggplot(data = data) +
  geom_bar(aes(device))

# Distribution of os
ggplot(data = data) +
  geom_bar(aes(os))

# Distribution of app
ggplot(data = data) +
  geom_bar(aes(app))

# Distribution of channel
ggplot(data = data) +
  geom_bar(aes(channel))

# Distribution of is_attributed (target)
ggplot(data = data) +
  geom_bar(aes(is_attributed))

# It is relevant to mention that the target variable is unbalanced.

# The data balance step will be carried out only in the train set, therefore,
# it will be applied following the partition of the data in train and test sets.

############################### DATA PARTIONING ################################

# Creating a random index to separate data into train (80%) and test (20%)
train_index <- createDataPartition(data$is_attributed, p = 0.8, list = FALSE)

# Creating train and test sets based on train.index
train = data[ train_index,]
test  = data[-train_index,]

# Removing data, train_index and free unused memory
rm(train_index)
rm(data)
invisible(gc())

############################# DATA BALANCING ###################################

# Let's try three different balancing techniques:

### undersampling;
### oversampling with package ROSE;
### mix of undersampling and oversampling.

# Checking unbalanced in train set
table(train$is_attributed)

# let's transform the click_time variable for factor type for balancing purposes,
# as the function ROSE cannot handle datetime type
train$click_time = as.factor(train$click_time)

### UNDERSAMPLING

# Separate majority and minority classes
train_majority = train[train$is_attributed==0]
train_minority = train[train$is_attributed==1]

# Creating a samples without replacement from train_majority based on the ip.
# The value XXX was select to produce a majority sample not too big compared
# to the minority sample
majority_under = sample(train_majority$ip, 100, replace=FALSE)

# Creating and filling in the indexes list for collecting all the repeated ips in
# the train set and considering the majority_under
j = 1
under_index = list()
for (i in 1:length(train$ip)) {
  if (train[i]$ip %in% majority_under) {
    under_index[[j]] = i
    j = j + 1}
}

# Creating the train_under with only the information of the majority_under
train_under = train[unlist(under_index), ]

# Binding the information of train_minority into the train_under
train_under = rbind(train_under, train_minority)

# Checking the distribution of target variable after balancing using the
# under sampling technique
table(train_under$is_attributed)

# Checking the data types of train_unde
glimpse(train_under)

### OVERSAMPLING

# Balancing the data using the package and function ROSE.
train_over = ROSE(is_attributed ~ ., data = train, N = 158000, seed=111)$data

# Checking the distribution of target variable after balancing using the
# oversampling technique using the function ROSE
table(train_over$is_attributed)

# Checking the data types of train_over
glimpse(train_over)

### UNDERSAMPLING AND OVERSAMPLING

# Based on the number of observations for is_attributed == 0 for train_under,
# an over sample based on the train_under set was created using ROSE, resulting
# in the train_underOver set
train_underOver = ROSE(is_attributed ~ ., data = train_under, N = 4000, seed=111)$data

# Checking the distribution of target variable after balancing using the
# under sampling followed by over sampling techniques for train_under
table(train_underOver$is_attributed)

# Transforming click_time of train_under, train_over and train_underOver back to
# datetime type
train_over$click_time = ymd_hms(train_over$click_time)
train_under$click_time = ymd_hms(train_under$click_time)
train_underOver$click_time = ymd_hms(train_underOver$click_time)

############################ FEATURE ENGINEERING ###############################

# Let's consider that a click is associated to an specific user, which
# needs an 'os' installed in a 'device' using a specific 'ip' address. So, let's
# create a variable named 'userID' to represent this.

# Let's consider the number of clicks in one hour per userID, ip, device, os, app, channel and
# create a few variables.

### day - day of the month that occurred an specific click;
### hour - hour of the day that occurred an specific click;
### minute - minute that occurred an specific click;
### userID_clicks_h - number of clicks per userID (ip + device + os) per hour;
### ip_clicks_h - number of clicks from a specific ip per hour;
### app_clicks_h - number of clicks in a specific app per hour;
### channel_clicks_h - number of clicks in a specific channel per hour;
### ip_app_clicks_h - number of clicks using a combination of ip and app per hour;
### ip_channel_clicks_h - number of clicks using a combination of ip and channel
# per hour.

# Creating a function to process the train and test (later on) sets
process_data <- function(df) {
  df = df %>%
    mutate(day = day(click_time),
           hour = hour(click_time),
           minute = minute(click_time)) %>%
    add_count(ip, device, os, day, hour) %>% rename(userID_clicks_h = n) %>%
    add_count(ip, day, hour) %>% rename(ip_clicks_h = n) %>%
    add_count(app, day, hour) %>% rename(app_clicks_h = n) %>%
    add_count(channel, day, hour) %>% rename(channel_clicks_h = n) %>%
    add_count(ip, app, day, hour) %>% rename(ip_app_clicks_h = n) %>%
    add_count(ip, channel, day, hour) %>% rename(ip_channel_clicks_h = n) %>%
    mutate(ip = NULL, click_time = NULL)
}

# Transform click_time of train set to datetime type
train$click_time = ymd_hms(train$click_time)

# Processing train
train = process_data(train)

# Processing train_over
train_over = process_data(train_over)

# Processing train_under
train_under = process_data(train_under)

# Processing train_underOver
train_underOver = process_data(train_underOver)

#############################  FEATURE SELECTION ###############################

# Let's check for correlation considering our new variables and target variable

# Transforming the target variable to integer only for correlation purposes
train$is_attributed = as.integer(train$is_attributed)
train_over$is_attributed = as.integer(train_over$is_attributed)
train_under$is_attributed = as.integer(train_under$is_attributed)
train_underOver$is_attributed = as.integer(train_underOver$is_attributed)

# Checking correlation on train
train_cor = cor(train[,c(5:14)])
train_cor
corrplot(train_cor)
# There was no strong correlation between target variable and predictors using
# the original train set without any balance.

# Checking correlation on train_over
train_over_cor = cor(train_over[,c(5:14)])
train_over_cor
corrplot(train_over_cor)
# In this case there appears to be a very strong linear correlation between target variable
# and userID_clicks_h, ip_clicks_h, ip_app_clicks_h and ip_channel_clicks_h
# Moreover, a relatively strong correlation between the target variable and
# app_clicks_h and channel_clicks_h was also observed.
# However, it is important to consider that most of those predictors are also
# correlated and should not be used altogether.

# Checking correlation on train_under_cor
train_under_cor = cor(train_under[,c(5:14)])
train_under_cor
corrplot(train_under_cor)
# No strong linear correlation was observed. However a few weak ones were observed
# In this case, let's select all variables and we can try different combinations but
# paying attention to the correlated predictors

# Checking correlation on train_under2_cor
train_underOver_cor = cor(train_underOver[,c(5:14)])
train_underOver_cor
corrplot(train_underOver_cor)
# For the train_underOver set, it seems that the correlations results lay in
# between the previous results. There are a few strong correlations observed,
# but less strong compared to the train_over set.
# Let's try a few combination of predictors taking care with the correlated ones

# Transforming the target variable back to factor
train$is_attributed = as.factor(train$is_attributed)
train_over$is_attributed = as.factor(train_over$is_attributed)
train_under$is_attributed = as.factor(train_under$is_attributed)
train_underOver$is_attributed = as.factor(train_underOver$is_attributed)

# Removing objects that are not need anymore and free unused memory
rm(i, j, majority_under, train_majority, train_minority, under_index)
invisible(gc())
# Now, before we go for the models' creation, let's do the same transformations
# that were done for train sets for the test set as well.

############################# PROCESSING TEST ##################################

# Processing test
test = process_data(test)

# Checking data types in test
glimpse(test)

# Alright, all the processing is done, let's create a few models to perform
# the model selection.

########################### MODEL SELECTION ####################################

# Let's try two classification algorithms to create the models and identify
# the train sets that results in better accuracy. Then we can concentrate our
# final steps for model creation using the chosen train set.

# - Random Forest
# - Naive Bayes

###### RANDOM FOREST ######

# Checking the number of categories, considering each variable
apply(X = train, MARGIN = 2, FUN = function(x) {length(unique(x))})
# With Random Forest can only handle 53 categories per variable, therefore the
# categorical variables will be not included in the random forest versions

##### TRAIN (non-balanced train set) #####

# Re-checking correlation of train
train_cor

# Model creation
rf1 = randomForest(is_attributed ~ hour + userID_clicks_h + app_clicks_h + 
                     channel_clicks_h + ip_app_clicks_h + ip_channel_clicks_h,
                   data = train, importance = TRUE)

# Visualizing the results of model
rf1

# Visualizing the importance of variables
varImpPlot(rf1)

# Creating a vector to hold the observed values for the test set
obs = test$is_attributed
obs_oL = ifelse(obs == 0, 1, 2)
obs_oL = as.factor(obs_oL)

# Prediction using test data
pred_rf1 = predict(rf1, test[,-5])

# Calculating accuracy for model
accuracy_rf1 = sum(obs_oL == pred_rf1) / length(obs_oL)
accuracy_rf1

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf1)

# The results shows that this model did not do a good job. It mistakes all
# the results from is_attributed that had a much lower number of observations.
# That's why it is important to balance the data.

# Let's create a new version of the model using the four most important variables,
# according to MeanDecreaseGini's index.


# Model creation
rf2 = randomForest(is_attributed ~ hour + app_clicks_h + channel_clicks_h,
                   data = train, importance = TRUE)

# Visualizing the results of model
rf2

# Visualizing the importance of variables
varImpPlot(rf2)

# Prediction using test data
pred_rf2 = predict(rf2, test[,-5])

# Calculating accuracy for model
accuracy_rf2 = sum(obs_oL == pred_rf2) / length(obs_oL)
accuracy_rf2

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf2)

# Well, it did not work any better than the previous version. Let's check some
# models using the train_over set


##### TRAIN_OVER #####

# Re-checking correlation of train_over
train_over_cor

# Model creation
rf3 = randomForest(is_attributed ~ hour + userID_clicks_h + app_clicks_h + 
                     channel_clicks_h,
                   data = train_over, importance = TRUE)

# Visualizing the results of model
rf3

# Visualizing the importance of variables
varImpPlot(rf3)

# Prediction using test data
pred_rf3 = predict(rf3, test[,-5])

# Calculating accuracy of model
accuracy_rf3 = sum(obs_oL == pred_rf3) / length(obs_oL)
accuracy_rf3

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf3)

# The same problem as observed in rf1, where the model classifies all as the
# class that possess higher count


# Model creation
rf4 = randomForest(is_attributed ~ userID_clicks_h + app_clicks_h,
                   data = train_over, importance = TRUE)

# Visualizing the results of model
rf4

# Visualizing the importance of variables
varImpPlot(rf4)

# Prediction using test data
pred_rf4 = predict(rf4, test[,-5])

# Calculating accuracy of model
accuracy_rf4 = sum(obs_oL == pred_rf4) / length(obs_oL)
accuracy_rf4

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf4)
# The same problem as observed in rf1, where the model classifies all as the
# class that possess higher count

# It seems that we are having some overfitting problem with the over sample train
# set. Let's try the train_under set.


##### TRAIN_UNDER #####

# Re-checking correlation of train_over
train_under_cor

# Model creation
rf5 = randomForest(is_attributed ~ hour + ip_clicks_h +
                     app_clicks_h + channel_clicks_h,
                   data = train_under, importance = TRUE)

# Visualizing the results of model
rf5

# Checking variable importance
varImpPlot(rf5)

# Prediction using test data
pred_rf5 = predict(rf5, test[,-5])

# Calculating accuracy for model
accuracy_rf5 = sum(obs_oL == pred_rf5) / length(obs_oL)
accuracy_rf5

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf5)

# Although it still makes some mistakes, this model seems to be doing a
# much better job than the previous versions.

# Let's try another version, using the userID_clicks_h instead of ip_clicks_h


# Model creation
rf6 = randomForest(is_attributed ~ hour + userID_clicks_h +
                     app_clicks_h + channel_clicks_h,
                   data = train_under, importance = TRUE)

# Visualizing the results of model
rf6

# Checking variable importance
varImpPlot(rf6)

# Prediction using test data
pred_rf6 = predict(rf6, test[,-5])

# Calculating accuracy for model
accuracy_rf6 = sum(obs_oL == pred_rf6) / length(obs_oL)
accuracy_rf6

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf6)

# The version rf5 was better than the version rf6 and all the others tested
# until this point.

# Now, let's try the other train set >> train_underOver


##### TRAIN_UNDEROVER #####

# Re-checking correlation of train_over
train_underOver_cor

# Model creation
rf7 = randomForest(is_attributed ~ hour + ip_channel_clicks_h + app_clicks_h,
                   data = train_underOver, importance = TRUE)

# Visualizing the results of model
rf7

# Prediction using test data
pred_rf7 = predict(rf7, test[,-5])

# Calculating accuracy for model
accuracy_rf7 = sum(obs_oL == pred_rf7) / length(obs_oL)
accuracy_rf7

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf7)
# The same problem presente by rf1, the model classifies all as the class that
# possess higher count

# Let's try removing the predictor app_clicks_h


# Model creation
rf8 = randomForest(is_attributed ~ hour + ip_channel_clicks_h,
                   data = train_underOver, importance = TRUE)

# Visualizing the results of model
rf8

# Prediction using test data
pred_rf8 = predict(rf8, test[,-5])

# Calculating accuracy for model
accuracy_rf8 = sum(obs_oL == pred_rf8) / length(obs_oL)
accuracy_rf8

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf8)
# The same problem as before, the model classifies all as the class that
# possess higher count


# Model creation
rf9 = randomForest(is_attributed ~ hour + userID_clicks_h + app_clicks_h,
                   data = train_underOver, importance = TRUE)

# Visualizing the results of model
rf9

# Prediction using test data
pred_rf9 = predict(rf9, test[,-5])

# Calculating accuracy for model
accuracy_rf9 = sum(obs_oL == pred_rf9) / length(obs_oL)
accuracy_rf9

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf9)
# The same problem as before, the model classifies all as the class that
# possess higher count


# Model creation
rf10 = randomForest(is_attributed ~ hour + userID_clicks_h,
                    data = train_underOver, importance = TRUE)

# Visualizing the results of model
rf10

# Prediction using test data
pred_rf10 = predict(rf10, test[,-5])

# Calculating accuracy for model
accuracy_rf10 = sum(obs_oL == pred_rf10) / length(obs_oL)
accuracy_rf10

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_rf10)
# The same problem as before, the model classifies all as the class that
# possess higher count

# So, the undercsampling technique seems to have worked better, meaning that the
# selected sample was able to produce a more accurate model, at least for the
# random forest algorithm.

# Let's try some versions using the Naive Bayes algorithm


###### NAIVE BAYES ######

# For the Naive Bayes version, we can also try to include the categorical
# variables for testing

##### TRAIN (non-balanced train set) #####

# Model creation
nb1 = naiveBayes(is_attributed ~ app + device + os + channel + hour +
                   userID_clicks_h + app_clicks_h + channel_clicks_h +
                   ip_app_clicks_h + ip_channel_clicks_h,
                 data = train, importance = TRUE)

# Visualizing the results of model
nb1

# Prediction using test data
pred_nb1 = predict(nb1, test[,-5])

# Calculating accuracy for model
accuracy_nb1 = sum(obs_oL == pred_nb1) / length(obs_oL)
accuracy_nb1

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb1)

# This model looks much better than the previous ones and it seems that we
# have improved compared to the best version of random forest (rf5)

# Let's try a second version as well but let's keep this model to be further
# adjusted later on.


# Model creation
nb2 = naiveBayes(is_attributed ~ app + device + os + channel + hour +
                   app_clicks_h + channel_clicks_h,
                 data = train, importance = TRUE)

# Visualizing the results of model
nb2

# Prediction using test data
pred_nb2 = predict(nb2, test[,-5])

# Calculating accuracy for model
accuracy_nb2 = sum(obs_oL == pred_nb2) / length(obs_oL)
accuracy_nb2

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb2)

# Well, it seems that we could improve a little bit compared to nb1.
# This algorithm seems to work very well with the original unbalanced dataset.
# Could be a good option.


##### TRAIN_OVER #####

# Model creation
# Model creation
nb3 = naiveBayes(is_attributed ~ app + device + os + channel + hour +
                   userID_clicks_h + app_clicks_h + 
                   channel_clicks_h,
                 data = train_over, importance = TRUE)

# Prediction using test data
pred_nb3 = predict(nb3, test[,-5])

# Calculating accuracy for model
accuracy_nb3 = sum(obs_oL == pred_nb3) / length(obs_oL)
accuracy_nb3

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb3)

# Here we are back to overfitting problem. Therefore, using the over sampling
# technique in this data set generated some problems.
# From now on, the train_over set will not be used anymore.


##### TRAIN_UNDER #####

# Model creation
nb4 = naiveBayes(is_attributed ~ app + device + os + channel +
                   hour + ip_clicks_h + app_clicks_h + channel_clicks_h,
                 data = train_under)

# Visualizing the results of model
nb4

# Prediction using test data
pred_nb4 = predict(nb4, test[,-5])

# Calculating accuracy for model
accuracy_nb4 = sum(obs_oL == pred_nb4) / length(obs_oL)
accuracy_nb4

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb4)

# This model did reasonably but worse than the models created with the train set

# Let's try a second version with the train_under set

# Model creation
nb5 = naiveBayes(is_attributed ~ app + device + os + channel +
                   hour + userID_clicks_h + app_clicks_h + channel_clicks_h,
                 data = train_under)

# Visualizing the results of model
nb5

# Prediction using test data
pred_nb5 = predict(nb5, test[,-5])

# Calculating accuracy for model
accuracy_nb5 = sum(obs_oL == pred_nb5) / length(obs_oL)
accuracy_nb5

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb5)

# It did not get any better. Let's try the train_underOver set



##### TRAIN_UNDEROVER #####

# Model creation
nb6 = naiveBayes(is_attributed ~ app + device + os + channel +
                   hour + ip_channel_clicks_h + app_clicks_h,
                 data = train_underOver)

# Visualizing the results of model
nb6

# Prediction using test data
pred_nb6 = predict(nb6, test[,-5])

# Calculating accuracy for model
accuracy_nb6 = sum(obs_oL == pred_nb6) / length(obs_oL)
accuracy_nb6

# Visualizing the Confusion Matrix for the observed and predicted values
table(obs_oL, pred_nb6)

# From the above analysis, the Naive Bayes algorithm did a much better job than
# the Random Forest algorithm. In the Naive Bayes, it was possible to include the
# categorical variables, which probably was one of the reasons for achieving 
# better results with this algorithm. Moreover, the Naive Bayes also handled very
# well the unbalanced original train set.

# Remove all the variables from the environment
rm(process_data, train_over, train_under, train_underOver,
   train_over_cor, train_under_cor, train_underOver_cor,
   rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10,
   nb1, nb2, nb3, nb4, nb5, nb6,
   pred_rf1, pred_rf2, pred_rf3, pred_rf4, pred_rf5,
   pred_rf6, pred_rf7, pred_rf8, pred_rf9, pred_rf10,
   pred_nb1, pred_nb2, pred_nb3, pred_nb4, pred_nb5, pred_nb6,
   accuracy_rf1, accuracy_rf2, accuracy_rf3, accuracy_rf4,
   accuracy_rf5, accuracy_rf6, accuracy_rf7, accuracy_rf8,
   accuracy_rf9, accuracy_rf10, accuracy_nb1, accuracy_nb2,
   accuracy_nb3, accuracy_nb4, accuracy_nb5, accuracy_nb6)

# Clear console
cat("\014")

# Clear plots and free unused memory
graphics.off()
invisible(gc())

# From now on, let's work with the original train set and test one more algorithm
# to create our final version of the model. Let's try the following algorithms:

# - Naive Bayes - already showed good results
# - SVM - linear and radial kernels

# Let's choose different sets of predictors to be used in different versions
# of the model.

# 1. app + device + os + channel + hour + userID_clicks_h + app_clicks_h + channel_clicks_h
# 2. app + device + os + channel + hour + userID_clicks_h + app_clicks_h
# 3. app + device + os + channel + hour + userID_clicks_h + channel_clicks_h
# 4. app + device + os + channel + hour + userID_clicks_h
# 5. app + device + os + channel + hour + app_clicks_h
# 6. app + device + os + channel + hour + channel_clicks_h

# 7. app + channel + hour + userID_clicks_h + app_clicks_h + channel_clicks_h
# 8. app + channel + hour + userID_clicks_h + app_clicks_h
# 9. app + channel + hour + userID_clicks_h + channel_clicks_h
# 10. app + channel + hour + userID_clicks_h

# 11. app + device + os + channel + hour + ip_clicks_h + app_clicks_h + channel_clicks_h
# 12. app + device + os + channel + hour + ip_clicks_h + app_clicks_h
# 13. app + device + os + channel + hour + ip_clicks_h + channel_clicks_h
# 14. app + device + os + channel + hour + ip_clicks_h
# 15. app + device + os + channel + hour + app_clicks_h
# 16. app + device + os + channel + hour + channel_clicks_h

# 17. app + channel + hour + ip_clicks_h + app_clicks_h + channel_clicks_h
# 18. app + channel + hour + ip_clicks_h + app_clicks_h
# 19. app + channel + hour + ip_clicks_h + channel_clicks_h
# 20. app + channel + hour + ip_clicks_h


# In total, there will be created 60 model version, being 20 constructed by the
# algorithm Naive Bayes and 40 by the SVM, including 20 using the linear kernel
# and 20 using the radial kernel.

# First, let's create a list to hold all the combinations of predictors (20)
predictors_list = list(
  "is_attributed ~ app + device + os + channel + hour + userID_clicks_h + app_clicks_h + channel_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + userID_clicks_h + app_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + userID_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + app_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + channel_clicks_h",
  "is_attributed ~ app + channel + hour + userID_clicks_h + app_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + hour + userID_clicks_h + app_clicks_h",
  "is_attributed ~ app + channel + hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + hour + userID_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + ip_clicks_h + app_clicks_h + channel_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + ip_clicks_h + app_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + ip_clicks_h + channel_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + ip_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + app_clicks_h",
  "is_attributed ~ app + device + os + channel + hour + channel_clicks_h",
  "is_attributed ~ app + channel + hour + ip_clicks_h + app_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + hour + ip_clicks_h + app_clicks_h",
  "is_attributed ~ app + channel + hour + ip_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + hour + ip_clicks_h"
)


###### NAIVE BAYES ######

# Creating four lists to hold the results
nb_model_list = list()
nb_pred_list = list()
nb_accuracy_list = list()
nb_tables_list = list()

# Iterating throught the predictors_list to generate models
for (i in 1:length(predictors_list)) {
  
  # Model creation and filling in the nb_model_list
  nb_model_list[[i]] = naiveBayes(as.formula(predictors_list[[i]]),
                                  data = train)
  
  # Prediction using test data and filling in the nb_pred_list
  nb_pred_list[[i]] = predict(nb_model_list[[i]], test[,-5])
  
  # Calculating accuracy for model and filling in the nb_accuracy_list
  nb_accuracy_list[[i]] = sum(obs_oL == nb_pred_list[[i]]) / length(obs_oL)
  
  # Filling the list nb_tables_list with the confusion matrix of results
  nb_tables_list[[i]] = table(obs_oL, nb_pred_list[[i]])
  print(paste("Step", i, "of 20!!"))
}

# Free unused memory
invisible(gc())


###### SVM - linear kernel ######

# Creating four lists to hold the results
svm1_model_list = list()
svm1_pred_list = list()
svm1_accuracy_list = list()
svm1_tables_list = list()


# Iterating throught the predictors_list to generate models
for (i in 1:length(predictors_list)) {
  
  # Model creation and filling in the svm1_model_list
  svm1_model_list[[i]] = svm(as.formula(predictors_list[[i]]),
                             data = train,
                             type = "C-classification",
                             kernel = "linear")
  
  # Prediction using test data and filling in the svm1_pred_list
  svm1_pred_list[[i]] = predict(svm1_model_list[[i]], test[,-5])
  
  # Calculating accuracy for model and filling in the svm1_accuracy_list
  svm1_accuracy_list[[i]] = sum(obs_oL == svm1_pred_list[[i]]) / length(obs_oL)
  
  # Filling the list svm1_tables_list with the confusion matrix of results
  svm1_tables_list[[i]] = table(obs_oL, svm1_pred_list[[i]])
  print(paste("Step", i, "of 20!!"))
}

# Free unused memory
invisible(gc())


###### SVM - radial kernel ######

# Creating four lists to hold the results
svm2_model_list = list()
svm2_pred_list = list()
svm2_accuracy_list = list()
svm2_tables_list = list()

# Iterating throught the predictors_list to generate models
for (i in 1:length(predictors_list)) {
  
  # Model creation and filling in the svm1_model_list
  svm2_model_list[[i]] = svm(as.formula(predictors_list[[i]]),
                             data = train,
                             type = "C-classification",
                             kernel = "radial")
  
  # Prediction using test data and filling in the svm1_pred_list
  svm2_pred_list[[i]] = predict(svm2_model_list[[i]], test[,-5])
  
  # Calculating accuracy for model and filling in the svm1_accuracy_list
  svm2_accuracy_list[[i]] = sum(obs_oL == svm2_pred_list[[i]]) / length(obs_oL)
  
  # Filling the list svm1_tables_list with the confusion matrix of results
  svm2_tables_list[[i]] = table(obs_oL, svm2_pred_list[[i]])
  print(paste("Step", i, "of 20!!"))
}

# Free unused memory
invisible(gc())

# Let's visualize the accuracy and confusion matrix of the models:
for (i in 1:length(nb_accuracy_list)) {
  print(paste("Accuracy for version", i, "- NB:", nb_accuracy_list[[i]]))
  print(paste("Accuracy for version", i, "- SVM_linear:", svm1_accuracy_list[[i]]))
  print(paste("Accuracy for version", i, "- SVM_radial:", svm2_accuracy_list[[i]]))
  print(paste("Confusion matrix for version", i, "- NB:", nb_tables_list[[i]]))
  print(paste("Confusion matrix for version", i, "- SVM_linear:", svm1_tables_list[[i]]))
  print(paste("Confusion matrix for version", i, "- SVM_radial:", svm2_tables_list[[i]]))
}

# First of all, all versions of SVM considering the radial kernel produced models
# that were probably over fitted and did not work at all with the test data. Althoug
# it presented a high level o accuracy (approx. 99.8%), it was not able to classify
# any register into one class (download the app >>> is_attributed = 1)

# In the case of SVM using the linear kernel, most of models presented similar results,
# with very high accuracy but a lot of wrong classification for one of the classes,
# which is the one representing that an user did download the app >> is_attributed = 1

# The best versions of the model were created with the Naive Bayes algorithm. The
# accuracy varied from 98.25% (version 7) to 99.10% (versions 10 and 20). Let's
# check these versions a bit further

# Creating two new lists to hold the errors (typeI and typeII)
error_typeI = list()
error_typeII = list()

# Iterating through nb_tables_list and filling the errors lists
for (i in 1:length(nb_tables_list)) {
  error_typeI[[i]] = round(nb_tables_list[[i]][3])
  error_typeII[[i]] = round(nb_tables_list[[i]][2])
}

# Creating a data.frame with errors, model versions and accuracies.
errors = data.frame(error_typeI = unlist(error_typeI),
                    error_typeII = unlist(error_typeII),
                    model = sapply(c(1:20), function(x) {paste0("Version-",x)}),
                    accuracy = sapply(c(1:20), function(x) {round(nb_accuracy_list[[x]] * 100, 2)}))

# Sorting the data.frame based on error_typeI and error_typeII
errors %>%
  arrange(error_typeI, error_typeII)

# According to this problem, I guess it is better to classify a user as an user
# that will not download the app by mistake than the ones that will do because
# if someone that the model classified as they would not download the app, in
# the worst case scenario they will do the download. However, if I classify an
# user as potential "good" user and they don't download and keep on causing
# more clicks is worse. In summary we want a model with high sensitivity and high
# specificity, however it is better to gain in sensitivity, not losing too much
# specificity. Therefore, my choice is the model version 9, which achieves
# a very low rate of error for the first case mentioned above, while achieving
# a still high rate of error, but lower than most of the model versions created
# in this project. Moreover, it achieved a very good accuracy of 98.99%.

# Now, let's try to improve it a little bit further if possible.

################################ MODEL TUNNING #################################

# ##Predictors used in version 9

#app + channel + hour + userID_clicks_h + channel_clicks_h

# Let's create a few variations just to make sure we still have the best features

#1. is_attributed ~ app + channel + hour + userID_clicks_h + channel_clicks_h **BASE
#2. is_attributed ~ app + channel + hour + userID_clicks_h
#3. is_attributed ~ app + channel + hour + channel_clicks_h
#4. is_attributed ~ app + channel + userID_clicks_h + channel_clicks_h
#5. is_attributed ~ app + channel + userID_clicks_h
#6. is_attributed ~ app + channel + channel_clicks_h
#7. is_attributed ~ app + hour + userID_clicks_h + channel_clicks_h
#8. is_attributed ~ app + hour + userID_clicks_h
#9. is_attributed ~ app + hour + channel_clicks_h
#10. is_attributed ~ channel + hour + userID_clicks_h + channel_clicks_h
#11. is_attributed ~ channel + hour + userID_clicks_h
#12. is_attributed ~ channel + hour + channel_clicks_h
#13. is_attributed ~ hour + userID_clicks_h + channel_clicks_h
#14. is_attributed ~ hour + userID_clicks_h
#15. is_attributed ~ app + userID_clicks_h + channel_clicks_h
#16. is_attributed ~ hour + channel_clicks_h
#17. is_attributed ~ app + userID_clicks_h
#18. is_attributed ~ app + channel_clicks_h
#19. is_attributed ~ channel + userID_clicks_h + channel_clicks_h
#10. is_attributed ~ channel + userID_clicks_h
#21. is_attributed ~ channel + channel_clicks_h

# Creating a list to hold all the combinations of predictors (21) for this step
predictors_list2 = list(
  "is_attributed ~ app + channel + hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + hour + userID_clicks_h",
  "is_attributed ~ app + channel + hour + channel_clicks_h",
  "is_attributed ~ app + channel + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + channel + userID_clicks_h",
  "is_attributed ~ app + channel + channel_clicks_h",
  "is_attributed ~ app + hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + hour + userID_clicks_h",
  "is_attributed ~ app + hour + channel_clicks_h",
  "is_attributed ~ channel + hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ channel + hour + userID_clicks_h",
  "is_attributed ~ channel + hour + channel_clicks_h",
  "is_attributed ~ hour + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ hour + userID_clicks_h",
  "is_attributed ~ hour + channel_clicks_h",
  "is_attributed ~ app + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ app + userID_clicks_h",
  "is_attributed ~ app + channel_clicks_h",
  "is_attributed ~ channel + userID_clicks_h + channel_clicks_h",
  "is_attributed ~ channel + userID_clicks_h",
  "is_attributed ~ channel + channel_clicks_h"
)

# Creating four lists to hold the results
nb_model_list2 = list()
nb_pred_list2 = list()
nb_accuracy_list2 = list()
nb_tables_list2 = list()

# Iterating throught the predictors_list to generate models
for (i in 1:length(predictors_list2)) {
  
  # Model creation and filling in the nb_model_list
  nb_model_list2[[i]] = naiveBayes(as.formula(predictors_list2[[i]]),
                                  data = train)
  
  # Prediction using test data and filling in the nb_pred_list
  nb_pred_list2[[i]] = predict(nb_model_list2[[i]], test[,-5])
  
  # Calculating accuracy for model and filling in the nb_accuracy_list
  nb_accuracy_list2[[i]] = sum(obs_oL == nb_pred_list2[[i]]) / length(obs_oL)
  
  # Filling the list nb_tables_list with the confusion matrix of results
  nb_tables_list2[[i]] = table(obs_oL, nb_pred_list2[[i]])
  print(paste("Step", i, "of 20!!"))
}

# Creating two new lists to hold the errors (typeI and typeII)
error2_typeI = list()
error2_typeII = list()

# Iterating through nb_tables_list and filling the errors lists
for (i in 1:length(nb_tables_list2)) {
  error2_typeI[[i]] = round(nb_tables_list2[[i]][3])
  error2_typeII[[i]] = round(nb_tables_list2[[i]][2])
}

# Creating a data.frame with errors, model versions and accuracies.
errors2 = data.frame(error2_typeI = unlist(error2_typeI),
                    error2_typeII = unlist(error2_typeII),
                    model = sapply(c(1:21), function(x) {paste0("Version-",x)}),
                    accuracy = sapply(c(1:21), function(x) {round(nb_accuracy_list2[[x]] * 100, 2)}))

# Sorting the data.frame based on error_typeI and error_typeII
errors2 %>%
  arrange(error2_typeI, error2_typeII)

# According to the results, the best option for our problem would be the version 1,
# which, in fact, exactly the same as version 9 from previous round.

# Therefore, this was the chosen final version for our problem.

# So, the chosen model, based on the accuracy evaluation is:
print(errors[9,])

# Let's print out some evaluation parameters for the chosen model
confusionMatrix(nb_pred_list[[9]], obs_oL)

################################## THE END #####################################

