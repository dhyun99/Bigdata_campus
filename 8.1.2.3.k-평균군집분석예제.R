
# k-평균 군집분석 수행하기
library(cclust)
set.seed(123)
cc.fit <- cclust(Nclus, 4, 20, method = "kmean")
ls(cc.fit)
# [1] "call"        "centers"     "changes"     "cluster"     "dist"        "initcenters"
# [7] "iter"        "method"      "ncenters"    "rate.method" "rate.par"    "size"       
#[13] "withinss"  


# k-평균 군집분석 결과 시각화하기
plot(Nclus, col = cc.fit$cluster,
     pch = ifelse(cc.fit$cluster == 1, 1,
              ifelse(cc.fit$cluster == 2, 2,
                  ifelse(cc.fit$cluster == 3, 5, 22))), cex = 1)
points(cc.fit$centers, col = 1:4, pch = c(16, 17, 18, 15), cex = 4)


# cluster::clusplot() 함수로 k-평균 군집분석 결과 시각화하기
library(cluster)
clusplot(Nclus, cc.fit$cluster)

