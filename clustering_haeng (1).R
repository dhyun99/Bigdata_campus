library(tidyverse)
library(factoextra)
library(clustMixType)
library(rlang)
library(ggplot2)
library(caret)
library(rpart.plot)
library(cluster)
library(NbClust)
library(textshape)
library(fpc)
library(dbscan)
library(fclust)
library(mclust)
library(Gmedian)

#Set working directory
setwd("E:/Preprocessing")

#Data Load
data <- read.csv("seoul_delivery_haeng.csv")

#Check data
glimpse(data)
summary(data)

#Set seed
set.seed(2021)

#Data Preprocessing
data$Haeng <- as.character(data$Haeng)
data<-data %>% remove_rownames %>% column_to_rownames(var = 'Haeng') %>% as.data.frame()

#NA check
colSums(is.na(data))

#Scale Model
scale_model <- caret::preProcess(data, method = c("center", "scale"))
data_stand <- predict(scale_model, data)

colSums(is.na(data_stand))
str(data_stand)

#Find the optimal number of clusters using Total within-cluster sum of squares
tot_withinss <- c()

for(i in 1:20){
  set.seed(2021)
  kmeans_cluster <- kmeans(data_stand, centers = i, iter.max = 10000)
  tot_withinss[i] <- kmeans_cluster$tot.withinss
}

plot(c(1:20), tot_withinss, type = "b",
     main = "Optimal number of clusters",
     xlab = "Number of clusters",
     ylab = "Total within-cluster sum of squares")


#F-test
r2 <- c()

for(i in 1:20)
{
  set.seed(2021)
  kmeans_cluster <- kmeans(data_stand, centers = i, iter.max = 10000)
  r2[i] <- kmeans_cluster$betweenss / kmeans_cluster$totss
}

plot(c(1:20), r2, type = "b",
     main = "The Elbow Method - Percentage of Variance Explained",
     xlab = "Number of clusters",
     ylab = "Percentage of Variance Explained")

#Silhouette Method
set.seed(2021)
km.res <- kmeans(data_stand, centers = 6)

sil <- silhouette(km.res$cluster, dist(data_stand))
fviz_silhouette(sil)

##K-means clustering
seoul_kmeans <- kmeans(data_stand, 6, nstart = 1000)
seoul_kmeans
table(seoul_kmeans$cluster)
seoul_kmeans$centers

fviz_cluster(seoul_kmeans, data = data_stand)



##K-Centroid Clustering(K-Medoids)
set.seed(2021)

##Find K in K-medoids
fviz_nbclust(data_stand, pam, method = 'wss')
fviz_nbclust(data_stand, pam, method = 'silhouette')

gap_stat <- clusGap(data_stand,
                    FUN = pam,
                    K.max = 10,
                    B = 50)

fviz_gap_stat(gap_stat)


pam_result <- pam(data_stand, k = 6)
table(pam_result$clustering)
pam_result$medoids
fviz_cluster(pam_result, data = data_stand)

##Gausian Clustering
BIC <- mclustBIC(data_stand)
plot(BIC)
summary(BIC)


mod1 <- Mclust(data_stand, x = BIC)
summary(mod1, parameters = T)


plot(mod1, what = 'classification')
table(rownames(data_stand), mod1$classification)
plot(mod1, what = 'uncertainty')


ICL <- mclustICL(data_stand)
summary(ICL)

LRT <- mclustBootstrapLRT(data_stand, modelName = "VEV")
LRT


a3mod <- Mclust(data_stand)
summary(a3mod, parameters = T)

Gausian <- a3mod$classification
Gausian


#DBSCAN
set.seed(2021)

dbscan::kNNdistplot(data_stand, k = 7)
abline(h = 2, lty =2)

db <- dbscan::dbscan(data_stand, eps = 1.7, MinPts = 2)
fviz_cluster(db, data_stand, stand = FALSE, frame = FALSE, geom = "point")