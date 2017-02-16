library(data.table)
library(SlopeOne)
rating_train<-read.csv("train-final.csv",header = TRUE)

rating_test<-read.csv("Test-final.csv",header = TRUE)

names(rating_train) <- c("user_id", "item_id", "rating")

names(rating_test) <- c("user_id", "item_id", "rating")

rating_train <- data.table(rating_train)

rating_test <- data.table(rating_test)

rating_train[, user_id := as.character(user_id)]
rating_train[, item_id := as.character(item_id)]
rating_test[, user_id := as.character(user_id)]
rating_test[, item_id := as.character(item_id)]

setkey(rating_train, user_id, item_id)
setkey(rating_test, user_id, item_id)

set.seed(1)

ratings_train_norm <- normalize_ratings(rating_train)

model <- build_slopeone(ratings_train_norm$ratings)

predictions <- predict_slopeone(model, 
                                rating_test, 
                                ratings_train_norm$ratings)


unnormalized_predictions <- unnormalize_ratings(normalized = ratings_train_norm,ratings = predictions)

rmse_slopeone <- sqrt(mean((unnormalized_predictions$predicted_rating - ratings_test$rating) ^ 2))


1.428223