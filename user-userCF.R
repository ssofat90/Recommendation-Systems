library(recommenderlab)
library(reshape2)
library(ggplot2)

tr<-read.csv("ratings.csv",header=TRUE)
g<-acast(tr, userId ~ movieId,value.var = "rating")
head(tr)

write.csv(train,"t.csv")

affinity.matrix<- as(g,"realRatingMatrix")


Rec.model=Recommender(affinity.matrix,method="UBCF", 
                      param=list(normalize = "Z-score",method="Cosine",nn=100, minRating=1))

recommended.items.u1 <- predict(Rec.model, affinity.matrix[1:5],type="ratings")
# to display them
as(recommended.items.u1, "matrix")[,1:5]
# to obtain the top 3
recommended.items.u1.top3 <- bestN(recommended.items.u1,n=3)
# to display them
as(recommended.items.u1.top3, "list")


# create evaluation scheme splitting taking 90% of the date for training and leaving 10% for validation or test
e <- evaluationScheme(affinity.matrix[1:671], method="split", train=0.8,given=20)
# creation of recommender model based on ubcf

Rec.ubcf <- Recommender(getData(e, "train"), method = "UBCF",param=list(normalize = "center", method="Cosine", nn=100))
# making predictions on the test data set
p.ubcf <- predict(Rec.ubcf, getData(e, "known"), type="ratings")
# obtaining the error metrics for both approaches and comparing them
error.ubcf<-calcPredictionAccuracy(p.ubcf, getData(e, "unknown"))

error.ubcf