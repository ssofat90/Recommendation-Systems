library(recommenderlab)
library(reshape2)
library(ggplot2)

tr<-read.csv("ratings.csv",header=TRUE)
g<-acast(tr, userId ~ movieId,value.var = "rating")

affinity.matrix<- as(g,"realRatingMatrix")

Rec.model=Recommender(affinity.matrix,method="IBCF", 
                      param=list(normalize = "Z-score",method="Cosine",k=100))

prediction <- predict(Rec.ibcf, affinity.matrix[1:5], type="ratings")
as(prediction, "matrix")[,1:5]


# create evaluation scheme splitting taking 80% of the date for training and leaving 20% for validation or test
e <- evaluationScheme(affinity.matrix, method="split", train=0.8,given=20)

# creation of recommender model based on ibcf for comparison
Rec.ibcf <- Recommender(getData(e, "train"), method= "IBCF",param=list(normalize = "center", method="Cosine", k=100))
# making predictions on the test data set
p.ibcf <- predict(Rec.ibcf, getData(e, "known"), type="ratings")
error.ibcf<-calcPredictionAccuracy(p.ibcf, getData(e, "unknown"))

error <- rbind(error.ubcf,error.ibcf)
rownames(error) <- c("UBCF","IBCF")
error
