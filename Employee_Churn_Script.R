#####Packages+working directory#####
setwd("C:/Users/simon/Desktop/2. semester/Data Mining for Business Decisions/R - Data Mining for Business Decisions")
packages_needed <- c('caret','tidyverse','rsample',
                     'visdat','recipes', 'dplyr', 'ggplot2', 'faraway', 'dummies',
                     'reshape2', 'vip', 'DataExplorer', 'skimr', 'pdp', 'pROC', 
                     'ROCR', 'corrplot', 'lubridate', 'glmnet', 'Matrix')
for (i in packages_needed){
  if (!require(i, character.only=TRUE)) install.packages(i, repos = "http://cran.us.r-project.org", quiet=TRUE)
  require(i, character.only=TRUE)
}

data <- read_csv("Data/HRDataset_v13.csv", 
                          col_types = cols(DeptID = col_skip(), 
                                           EmpID = col_skip(), EmpStatusID = col_skip(), 
                                           Employee_Name = col_skip(), FromDiversityJobFairID = col_skip(), 
                                           GenderID = col_skip(), HispanicLatino = col_skip(), 
                                           ManagerID = col_skip(), MaritalStatusID = col_skip(), 
                                           MarriedID = col_skip(), PerfScoreID = col_skip(), 
                                           PositionID = col_skip(), Termd = col_skip(), 
                                           Zip = col_skip(), DOB = col_skip(), LastPerformanceReview_Date = col_skip(),
                                           DateofTermination  = col_skip(), DateofHire  = col_skip()))


data$Position  <- as.factor(data$Position)
data$State <- as.factor(data$State)
data$Sex <- as.factor(data$Sex)
data$MaritalDesc <- as.factor(data$MaritalDesc)
data$CitizenDesc <- as.factor(data$CitizenDesc)
data$RaceDesc <- as.factor(data$RaceDesc)
data$TermReason <- as.factor(data$TermReason)
data$ManagerName <- as.factor(data$ManagerName)
data$RecruitmentSource <- as.factor(data$RecruitmentSource)
data$PerformanceScore <- as.factor(data$PerformanceScore)
data$Department <- as.factor(data$Department)
data$SpecialProjectsCount <- as.factor(data$SpecialProjectsCount)
data$EmpSatisfaction <- as.factor(data$EmpSatisfaction)

#####changing DOB to AGE at hiring variable:#####
#' Calculate age
#' 
#' By default, calculates the typical "age in years", with a
#' \code{floor} applied so that you are, e.g., 5 years old from
#' 5th birthday through the day before your 6th birthday. Set
#' \code{floor = FALSE} to return decimal ages, and change \code{units}
#' for units other than years.
#' @param dob date-of-birth, the day to start calculating age.
#' @param age.day the date on which age is to be calculated.
#' @param units unit to measure age in. Defaults to \code{"years"}. Passed to \link{\code{duration}}.
#' @param floor boolean for whether or not to floor the result. Defaults to \code{TRUE}.
#' @return Age in \code{units}. Will be an integer if \code{floor = TRUE}.
#' @examples
#' my.dob <- as.Date('1983-10-20')
#' age(my.dob)
#' age(my.dob, units = "minutes")
#' age(my.dob, floor = FALSE)


age <- function(dob, age.day = today(), units = "years", floor = TRUE) {
  calc.age = interval(dob, age.day) / duration(num = 1, units = units)
  if (floor) return(as.integer(floor(calc.age)))
  return(calc.age)
}

#changing DOB to AGE variable:
dataset$DOB <- as.character(dataset$DOB)
dataset$DOB <- as.Date(dataset$DOB)
dataset$DateofHire <- as.Date(dataset$DateofHire)

my.dob <- as.Date(dataset$DOB)
age(my.dob, age.day = as.Date(dataset$DateofHire))



plot_histogram(data)
#Create target variable
data <- data %>% 
    mutate(vol_term = ifelse(EmploymentStatus == "Voluntarily Terminated", "yes", "no"))

data <- data %>% 
  mutate(State = ifelse(State == "MA", "MA", "Other"))
data$State <- as.factor(data$State)


#Unorder the factor
data$vol_term <- as.factor(data$vol_term)
data <- data$vol_term %>% mutate_if(is.ordered, factor, ordered = FALSE)

# Reorder the factor so caret knows "Yes" is the positive class
data$vol_term <- factor(data$vol_term, levels = c("yes", "no"))



#####Removing terminated for a cause#####
data <- data %>% 
  filter(EmploymentStatus != "Terminated for Cause")

data <- data[,-c(8, 9, 17, 19, 20)] #removing employmentstatus, term reason, daycountpr, yearsemp & dayslatelast30
#####Data visualization/Exploration#####
summary(data)
view(data)
plot_bar(data)
plot_histogram(data)


#####Distribution of target variable#####
table(data$State) %>% 
  prop.table()


######NA's#####
vis_miss(data, cluster = TRUE, sort_miss=TRUE) 
plot_missing(data)

#####Sampling and partitioning######
set.seed(23)  # for reproducibility
split  <- initial_split(data, prop = 0.7, strata = vol_term) # doing a 70% / 30% split + stratifying
train  <- training(split)
test   <- testing(split)

#####Blueprint/Recipe#####
blueprint <- recipe(vol_term ~ ., data = train) %>%
  step_zv(all_predictors()) %>% #Both numeric and categorical
  step_nzv(all_predictors()) %>% #Both numeric and categorical
  step_mutate(PayRate = ifelse(PayRate >mean(PayRate), "Below average", "Above average")) %>%
  step_mutate(State = ifelse(State  == "MA", "MA", "other")) %>%
  step_mutate(age = ifelse(age > 35, "Below 35", "Above 35")) %>%
  step_mutate(EngagementSurvey = ifelse(EngagementSurvey > 3, "Below 3", "Above 3")) %>%
  step_string2factor(EngagementSurvey) %>% 
  step_string2factor(age) %>% 
  step_string2factor(PayRate) %>% 
  step_string2factor(State) %>% 
  step_other(Position, threshold = 0.05, other = "other") %>%
  step_other(RaceDesc, threshold = 0.05, other = "other") %>% 
  step_other(PerformanceScore, threshold = 0.08, other = "Below average") %>% 
  step_other(RecruitmentSource, threshold = 0.05, other = "other") %>%
  step_other(SpecialProjectsCount, threshold = 0.1, other = "1 or more") %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = T)  #one_hot = F for dummy encoding

  
# Second, train the blueprint on training data
prepare <- prep(blueprint, training = train)
view(prepare$term_info)
prepare

# Lastly, we can apply our blueprint to new data
baked_train <- bake(prepare, new_data = train)
baked_test <- bake(prepare, new_data = test)
baked_train

####More Blueprint#####
blueprint1 <- recipe(vol_term ~ ., data = train) %>%
  step_discretize(all_numeric())
  step_zv(all_predictors()) %>% #Both numeric and categorical
  step_nzv(all_predictors()) %>% #Both numeric and categorical
  step_integer(EmpSatisfaction) %>% # Ordinally encode all quality features, which are on a 1-10 Likert scale.
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_other(Position, threshold = 0.05, 
             other = "other") %>%
  step_other(RaceDesc, threshold = 0.05, 
             other = "other") %>% 
  step_other(PerformanceScore, threshold = 0.08, 
             other = "Below average") %>% 
  step_other(RecruitmentSource, threshold = 0.05, 
             other = "other") %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)  #one_hot = F for dummy encoding

  
  # Second, train the blueprint on training data
  prepare <- prep(blueprint1, training = train)
  view(prepare$term_info)
  prepare
  
  # Lastly, we can apply our blueprint to new data
  baked_train <- bake(prepare, new_data = train)
  baked_test <- bake(prepare, new_data = test)
  baked_train
  

#####Model building#####

#####CV#####
tr_control <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           classProbs = TRUE,
                           summaryFunction = prSummary)

#####CV ONESE#####
tr_control_oneSE <- trainControl(method = "cv", number = 10,
                                 selectionFunction = "oneSE",
                                 classProbs = TRUE,                 
                                 summaryFunction = prSummary)
####Final model#####
fitControl <- trainControl(method = "none",
                           classProbs = TRUE,                 
                           summaryFunction = prSummary)


#selectionFunction can be used to supply a function to algorithmically determine the final model. 
#There are three existing functions in the package: best is chooses the largest/smallest value, oneSE attempts to capture 

#logistic regression model
#!!!!! NO PARAMETERS FOR THIS MODEL!!!!!!!!!
set.seed(23)
start_time <- Sys.time()
LR_fit <- train(
  blueprint, 
  data = train, 
  method = "glm",
  family = binomial,
  trControl = tr_control,
  metric = "AUC"
)
end_time <- Sys.time()
end_time - start_time


######Penalized model#####
set.seed(23)
penalized_mod <- train(
  blueprint, 
  data = train, 
  method = "glmnet",
  family = "binomial",
  trControl = tr_control, 
  tuneLength = 10,
  metric = "AUC"
)
penalized_mod$bestTune
plot(penalized_mod)
penalized_mod$results

######Penalized model ONESE#####
set.seed(23)
penalized_mod_ONESE <- train(
  blueprint, 
  data = train, 
  method = "glmnet",
  family = "binomial",
  trControl = tr_control, 
  #tuneLength = 10,
  metric = "AUC"
)

penalized_mod_ONESE$bestTune
plot(penalized_mod_ONESE)


#####Assesment using CV on the training set#####
resamps <- resamples(list(LR = LR_fit,
                          penalized = penalized_mod))
summary(resamps)

pred_class <- predict(LR_fit, train)
confusionMatrix(
  data = relevel(pred_class, ref = "yes"), 
  reference = relevel(train$vol_term, ref = "yes")
)

pred_class1 <- predict(penalized_mod, train)
confusionMatrix(
  data = relevel(pred_class1, ref = "yes"), 
  reference = relevel(train$vol_term, ref = "yes")
)

pred_class3 <- predict(penalized_mod_ONESE, train)
confusionMatrix(
  data = relevel(pred_class3, ref = "yes"), 
  reference = relevel(train$vol_term, ref = "yes")
)


set.seed(23)
start_time <- Sys.time()
Final_mod <- train(
  blueprint, 
  data = train, 
  method = "glm",
  family = binomial,
  trControl = fitControl,
  metric = "AUC"
)
end_time <- Sys.time()
end_time - start_time

summary(Final_mod)

######Penalized model#####
set.seed(23)
penalized_mod <- train(
  blueprint, 
  data = train, 
  method = "glmnet",
  family = "binomial",
  trControl = fitControl, 
  tuneLength = 10,
  metric = "AUC"
)
penalized_mod$bestTune
plot(penalized_mod)
penalized_mod$results

######Penalized model ONESE#####
set.seed(23)
penalized_mod_ONESE <- train(
  blueprint, 
  data = train, 
  method = "glmnet",
  family = "binomial",
  trControl = fitControl, 
  #tuneLength = 10,
  metric = "AUC"
)




#####TEST#####
pred_class <- predict(Final_mod, test)
confusionMatrix(
  data = relevel(pred_class, ref = "yes"), 
  reference = relevel(test$vol_term, ref = "yes", mode = "everything")
)

confusionMatrix(test$vol_term, data = pred_class, positive = "yes", mode = "everything")

pred_class1 <- predict(penalized_mod, test)
confusionMatrix(
  data = relevel(pred_class1, ref = "yes"), 
  reference = relevel(test$vol_term, ref = "yes", mode = "everything")
)


pred_pen <- predict(penalized_mod_ONESE, test)
confusionMatrix(
  data = relevel(pred_pen, ref = "yes"), 
  reference = relevel(test$vol_term, ref = "yes")
)


#####Variable importance plot#####
vip(Final_mod, num_features = 100)

####Direction of variables#####
numvar_LR <- coef(Final_mod$finalModel) # only the sign is of interest
view(numvar_LR)



