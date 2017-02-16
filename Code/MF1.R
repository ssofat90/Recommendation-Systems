install.packages("recommenderlab")
install.packages("recosystem")

library(devtools)
install_github(repo = "SlopeOne", username = "tarashnot")
install_github(repo = "SVDApproximation", username = "tarashnot")
library(recommenderlab)
library(recosystem)
library(SlopeOne)
library(SVDApproximation)

library(data.table)
library(RColorBrewer)
library(ggplot2)

library(foreign)

setwd("C:/Users/Shush/Desktop/FinalDataset/ml-latest-small")
ratings<-read.csv("C:/Users/Shush/Desktop/FinalDataset/ml-latest-small/ratings.csv",header=TRUE)
write.dta(ratings, file = file.path(tempdir(), "ratings.dta"))
ratings <- as.data.frame(ratings)
is.data.frame(ratings)

data(ratings)

set.seed(1)
in_train <- rep(TRUE, nrow(ratings))
in_train[sample(1:nrow(ratings), size = round(0.2 * length(unique(ratings$user)), 0) * 5)] <- FALSE

ratings_train <- ratings[(in_train)]
ratings_test <- ratings[(!in_train)]

ratings_train<-read.csv("train-final.csv")
ratings_test<-read.csv("Test-final.csv")


write.table(ratings_train, file = "trainset.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(ratings_test, file = "testset.txt", sep = " ", row.names = FALSE, col.names = FALSE)

r = Reco()

opts <- r$tune("trainset.txt", opts = list(dim = 4, lrate = 0.05,
                                  costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 50, nfold = 10, verbose = FALSE))

r$train("trainset.txt", opts = c(opts$min, nthread = 1, niter = 50, verbose = FALSE))

outfile = tempfile()

r$predict("testset.txt", out_pred = outfile)

scores_real <- read.table("testset.txt", header = FALSE, sep = " ")
scores_pred <- scan(outfile)


rmse_mf <- sqrt(mean((scores_real$V3-scores_pred) ^ 2))
rmse_mf


