###########################################################################################################################################################################    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# EXAMPLE CODE
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# INSTALL AND LOAD PACKAGES
# READ DATA
# DEFINE OBJECTS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet", "WeightedROC")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#Set working directory
setwd("filepath")

folds = 10

# Read data
D <- readRDS("train_data.rds")

# Also read in the training data split into 10 CV folds stratified on the outcome.
splt <- readRDS("splt.rds")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# CODE INDIVIDUAL MODELS
# PREDICT FROM INDIVIDUAL MODELS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# We are fitting individual algorithms on the training set, excluding the iith validation set
# For demonstration purposes we are including one specification of each kind of

set.seed(123)

# bayesglm with defaults
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~. -w,data=do.call(rbind,splt[-ii]),family="binomial", weights = w)) 
#random forest (ranger) 
m2 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~. -w, data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = T, case.weights = do.call(rbind,splt[-ii])$w))
#mean
m3 <- lapply(1:folds,function(ii) weighted.mean(rbindlist(splt[-ii])$ch_smmtrue, w = rbindlist(splt[-ii])$w))
#glm
m4 <- lapply(1:folds, function(ii) glm(ch_smmtrue~. -w, data=do.call(rbind,splt[-ii]), family="binomial", weights = w))
#glmnet - also vary lambdas? 
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
#also - we standardized variables before in preprocessing, so need to set standardize = FALSE 
m5 <- lapply(1:folds, function(ii) cv.glmnet(model.matrix(~-1 + ., do.call(rbind,splt[-ii])[,-1]), as.matrix(do.call(rbind,splt[-ii])[,1]), alpha = 0,   family="binomial", nlambda = 100, lambda = NULL, type.measure = "deviance", nfolds = 10, standardize = FALSE, weights = do.call(rbind,splt[-ii])$w))

## Now, obtain the predicted probability of the outcome for observation in the ii-th validation set
# bayesglm
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=do.call(rbind,splt[ii]),type="response"))
# ranger  
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=do.call(rbind,splt[ii])))
# mean
p3 <- lapply(1:folds, function(ii) rep(m3[[ii]], nrow(splt[[ii]])))
#glm  
p4 <- lapply(1:folds, function(ii) predict(m4[[ii]], newdata = do.call(rbind,splt[ii]), type="response"))
#glmnet 
p5 <- lapply(1:folds, function(ii) predict(m5[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-1]), s="lambda.min", type="response"))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMBINE PREDICTIONS FROM ABOVE MODELS INTO ONE DATAFRAME 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# update dataframe 'splt' so that column1 is the observed outcome (y) and subsequent columns contain predictions above

for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,1],
                   p1[[i]], #bayesglm
                   p2[[i]]$predictions, #ranger
                   p3[[i]], #mean
                   p4[[i]], #glm
                   p5[[i]], #glmnet
                   splt[[i]][,8]) #weights 
}
# view the first 6 observations in the first fold 
head(data.frame(splt[[1]]))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE CV RISK FOR EACH METHOD
# AVERAGE CV RISK OVER ALL 10 FOLDS TO GET 1 PERFORMANCE MEASURE PER ALGORITHM
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# First, calculate CV risk for each method for the ii-th validation set
# our loss function is the rank loss; so our risk is (1-AUC)
#	use the AUC() function with input as the predicted outcomes and 'labels' as the true outcomes
risk1<-lapply(1:folds,function(ii) 1-WeightedAUC(WeightedROC(splt[[ii]][,2], splt[[ii]][,1], splt[[ii]][,7])))    # CV-risk for bayesglm
risk2<-lapply(1:folds,function(ii) 1-WeightedAUC(WeightedROC(splt[[ii]][,3], splt[[ii]][,1], splt[[ii]][,7])))		# CV-risk for ranger 
risk3<-lapply(1:folds,function(ii) 1-WeightedAUC(WeightedROC(splt[[ii]][,4], splt[[ii]][,1], splt[[ii]][,7])))		# CV-risk for mean
risk4<-lapply(1:folds,function(ii) 1-WeightedAUC(WeightedROC(splt[[ii]][,5], splt[[ii]][,1], splt[[ii]][,7])))		# CV-risk for glm
risk5<-lapply(1:folds,function(ii) 1-WeightedAUC(WeightedROC(splt[[ii]][,6], splt[[ii]][,1], splt[[ii]][,7])))		# CV-risk for glmnet

# Next, average the estimated risks across the folds to obtain 1 measure of performance for each algorithm
a<-rbind(cbind("bayesglm",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("ranger",mean(do.call(rbind, risk2),na.rm=T)),
         cbind("mean",mean(do.call(rbind,risk3), na.rm=T)),
         cbind("glm",mean(do.call(rbind,risk4), na.rm=T)),
         cbind("glmnet",mean(do.call(rbind,risk5), na.rm=T)))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# ESTIMATE SUPERLEARNER WEIGHTS BY MINIMIIZNG 1-auc
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# here: estimate SL weights using the optim() function to minimize (1-AUC)
# turn splt into a data frame (X) and define names 
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","bayesglm","ranger","mean","glm", "glmnet","w")
head(X)

# Define the function we want to optimize (SL.r)
SL.r<-function(A, y, par, w){
  A<-as.matrix(A)
  names(par)<-c("bayesglm",
                "ranger",
                "mean",
                "glm",
                "glmnet")
  predictions <- crossprod(t(A),par)
  wroc <- WeightedROC(predictions, y, w)
  cvRisk <- 1 - WeightedAUC(wroc)
}


# Define bounds and starting values
# init should be 1/par, par where par is number of predictors (excluding y)
bounds = c(0, Inf)
init <- rep(1/5, 5)

# Optimize SL.r
fit <- optim(par=init, fn=SL.r, A=X[,2:6], y=X[,1], w=X[,7],
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit

# Normalize the coefficients and look at them
alpha<-fit$par/sum(fit$par)
options(scipen=999) #changes from scientific to numeric notation 
alpha
sum(alpha)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# RE-FIT ALL ALGORITHMS TO ORIGINAL DATA WITHOUT CROSS-VALIDATION
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

set.seed(123)

# bayesglm with defaults
n1<- bayesglm(formula=ch_smmtrue~. -w,data=D,family="binomial", weights = w) 
#random forest (ranger) with a range of tuning parameters
n2 <- ranger(ch_smmtrue~. -w, data=D, num.trees = 500, mtry = 2, min.node.size = 10, replace = T, case.weights = D$w)
#mean
n3 <- weighted.mean(D$ch_smmtrue, w = D$w)
#glm
n4 <- glm(ch_smmtrue~. -w, data=D, family="binomial", weights = w)
#glmnet
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
#also - we standardized variables before in preprocessing, so need to set standardize = FALSE 
n5 <- cv.glmnet(model.matrix(~-1 + ., D[,-1]), as.matrix(D[,1]), alpha = 0,   family="binomial", nlambda = 100, lambda = NULL, type.measure = "deviance", nfolds = 10, standardize = FALSE, weights = D$w)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREDICT PROBABILITIES FROM EACH FIT USING ALL TEST DATA 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
## Now, obtain the predicted probability of the outcome for observation in the ii-th validation set
#bayesglm 
p1 <- predict(n1,newdata=test,type="response")
#ranger 
p2<- predict(n2,data=test)
#mean 
p3<- rep(n3, nrow(test))
#glm
p4 <- predict(n4, newdata = test, type="response")
#glmnet
p5 <- predict(n5, newx = as.matrix(test[,-1]), s="lambda.min", type="response")

predictions <- cbind(p1, p2$predictions, p3, p4, p5)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAKE A WEIGHTED COMBINATION OF PREDICTIONS USING NNLS COEFFICIENTS AS WEIGHTS 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
predmat <- as.matrix(predictions)
alpha <- as.matrix(alpha)
y_pred <- predmat%*%alpha

p<-data.frame(y=test$ch_smmtrue,y_pred=y_pred)

# Generate predicted classifications based whichever thresholds you choose
p <- p %>% mutate(pred_class50 = ifelse(y_pred >= 0.5, 1, 0),
                  pred_class30 = ifelse(y_pred >= 0.3, 1, 0),
                  pred_class20 = ifelse(y_pred >= 0.2, 1, 0),
                  pred_class15 = ifelse(y_pred >= 0.15, 1, 0))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

