
# rattle 패키지와 wine 데이터셋 불러오기
library(rattle)
data("wine")
str(wine)
#'data.frame':	178 obs. of  14 variables:
#$ Type           : Factor w/ 3 levels "1","2","3": 1 1 1 1 1 1 1 1 1 1 ...
#$ Alcohol        : num  14.2 13.2 13.2 14.4 13.2 ...
#$ Malic          : num  1.71 1.78 2.36 1.95 2.59 1.76 1.87 2.15 1.64 1.35 ...
#$ Ash            : num  2.43 2.14 2.67 2.5 2.87 2.45 2.45 2.61 2.17 2.27 ...
#$ Alcalinity     : num  15.6 11.2 18.6 16.8 21 15.2 14.6 17.6 14 16 ...
#$ Magnesium      : int  127 100 101 113 118 112 96 121 97 98 ...
#$ Phenols        : num  2.8 2.65 2.8 3.85 2.8 3.27 2.5 2.6 2.8 2.98 ...
#$ Flavanoids     : num  3.06 2.76 3.24 3.49 2.69 3.39 2.52 2.51 2.98 3.15 ...
#$ Nonflavanoids  : num  0.28 0.26 0.3 0.24 0.39 0.34 0.3 0.31 0.29 0.22 ...
#$ Proanthocyanins: num  2.29 1.28 2.81 2.18 1.82 1.97 1.98 1.25 1.98 1.85 ...
#$ Color          : num  5.64 4.38 5.68 7.8 4.32 6.75 5.25 5.05 5.2 7.22 ...
#$ Hue            : num  1.04 1.05 1.03 0.86 1.04 1.05 1.02 1.06 1.08 1.01 ...
#$ Dilution       : num  3.92 3.4 3.17 3.45 2.93 2.85 3.58 3.58 2.85 3.55 ...
#$ Proline        : int  1065 1050 1185 1480 735 1450 1290 1295 1045 1045 ...

table(wine$Type)
#  1  2  3 
# 59 71 48 


# 설명변수 표준화하기
wine2 <- scale(wine[-1])


# 군집 수를 늘려가면서 군집 내 제곱합과 군집 간 제곱합을 계산하여 엘보우 값 찾아보기
library(foreach)
nc <- 1:15
nc.res <- foreach(i = nc, .combine = rbind) %do% {
   with.ss <- sum(kmeans(wine2, centers = i)$withinss)
   between.ss <- kmeans(wine2, centers = i)$betweenss
   return(data.frame(nc = i, wss = with.ss, bss = between.ss))
}
nc.res
#   nc       wss          bss
#1   1 2301.0000 3.183231e-12
#2   2 1649.4400 6.515600e+02
#3   3 1270.7491 1.030251e+03
#4   4 1174.0421 1.126958e+03
#5   5 1105.8442 1.205847e+03
#6   6 1058.8808 1.245271e+03
#7   7  997.3963 1.311571e+03
#8   8  975.3081 1.357504e+03
#9   9  899.0106 1.422021e+03
#10 10  866.4230 1.408479e+03
#11 11  811.9807 1.478023e+03
#12 12  811.5378 1.523008e+03
#13 13  792.6500 1.504608e+03
#14 14  739.6378 1.574646e+03
#15 15  704.9835 1.599953e+03

plot(nc.res$nc, nc.res$wss, type = "b", xlab = "Number of Clusters",
     ylab = "Within groups sum of squares")

plot(nc.res$nc, nc.res$bss, type = "b", xlab = "Number of Clusters",
     ylab = "Between-cluster sum of squares")


# 최적의 군집 수(k) 확인하기
library(NbClust)
set.seed(123)
nbc <- NbClust(wine2, min.nc = 2, max.nc = 15, method = "kmeans")
ls(nbc)
#[1] "All.CriticalValues" "All.index" "Best.nc" "Best.partition"

table(nbc$Best.nc[1,])
#0 1 2  3 10 12 14 15
#2 1 4 15  1  1  1  1

barplot(table(nbc$Best.nc[1, ]),
        xlab = "Number of Cluster",
        ylab = "Number of Criteris")


# 군집 수 k = 3으로 k-평균 군집분석 모델 생성 및 군집 결과 시각화하기
set.seed(123)
km.fit <- kmeans(wine2, centers = 3, nstart = 25)
plot(wine2, col = km.fit$cluster,
     pch = ifelse(km.fit$cluster == 1, 1, ifelse(km.fit$cluster == 2, 2, 6)), cex = 1)
points(km.fit$centers, col = 1:3, pch = c(1, 2, 6), cex = 3)


# 군집 결과의 추가 정보 확인하기
ls(km.fit)
#[1] "betweenss" "centers" "cluster" "ifault" "iter" "size"
#[7] "tot.withinss" "totss" "withinss"

km.fit$size
#[1] 62 51 65

km.fit$cluster
#[1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 … 이하 생략

round(km.fit$centers, 3)
#  Alcohol  Malic    Ash Alcalinity Magnesium Phenols Flavanoids Nonflavanoids Proanthocyanins
#1   0.833 -0.303  0.364     -0.608     0.576   0.883      0.975        -0.561           0.579
#2   0.164  0.869  0.186      0.523    -0.075  -0.977     -1.212         0.724          -0.778
#3  -0.923 -0.393 -0.493      0.170    -0.490  -0.076      0.021        -0.033           0.058
#   Color    Hue Dilution Proline
#1  0.171  0.473    0.777   1.122
#2  0.939 -1.162   -1.289  -0.406
#3 -0.899  0.461    0.270  -0.752


# 각 군집별로 변수의 평균값을 측정 단위의 척도로 나타내기
aggregate(wine[-1], by = list(cluster = km.fit$cluster), FUN = mean)
#  cluster  Alcohol    Malic      Ash Alcalinity Magnesium  Phenols Flavanoids Nonflavanoids
#1       1 13.67677 1.997903 2.466290   17.46290 107.96774 2.847581  3.0032258     0.2920968
#2       2 13.13412 3.307255 2.417647   21.24118  98.66667 1.683922  0.8188235     0.4519608
#3       3 12.25092 1.897385 2.231231   20.06308  92.73846 2.247692  2.0500000     0.3576923
#  Proanthocyanins    Color       Hue Dilution   Proline
#1        1.922097 5.453548 1.0654839 3.163387 1100.2258
#2        1.145882 7.234706 0.6919608 1.696667  619.0588
#3        1.624154 2.973077 1.0627077 2.803385  510.1692


# wine 데이터에 군집 분류 결과의 cluster 속성 추가하기
wine$cluster <- km.fit$cluster
head(wine)
#  Type Alcohol Malic  Ash Alcalinity Magnesium Phenols Flavanoids Nonflavanoids Proanthocyanins
#1    1   14.23  1.71 2.43       15.6       127    2.80       3.06          0.28            2.29
#2    1   13.20  1.78 2.14       11.2       100    2.65       2.76          0.26            1.28
#3    1   13.16  2.36 2.67       18.6       101    2.80       3.24          0.30            2.81
#4    1   14.37  1.95 2.50       16.8       113    3.85       3.49          0.24            2.18
#5    1   13.24  2.59 2.87       21.0       118    2.80       2.69          0.39            1.82
#6    1   14.20  1.76 2.45       15.2       112    3.27       3.39          0.34            1.97
#  Color  Hue Dilution Proline cluster
#1  5.64 1.04     3.92    1065       1
#2  4.38 1.05     3.40    1050       1
#3  5.68 1.03     3.17    1185       1
#4  7.80 0.86     3.45    1480       1
#5  4.32 1.04     2.93     735       1
#6  6.75 1.05     2.85    1450       1


table(wine$Type)
# 1  2  3
#59 71 48

table(wine$cluster)
# 1  2  3 
#62 51 65 


# wine 데이터에서 cluster, Type 변수의 그룹으로 데이터(레코드) 수 출력하기
aggregate(wine, by = list(cluster = wine$cluster, Type = wine$Type), NROW)
#  cluster Type Type Alcohol Malic Ash Alcalinity Magnesium Phenols Flavanoids Nonflavanoids
#1       1    1   59      59    59  59         59        59      59         59            59
#2       1    2    3       3     3   3          3         3       3          3             3
#3       2    2    3       3     3   3          3         3       3          3             3
#4       3    2   65      65    65  65         65        65      65         65            65
#5       2    3   48      48    48  48         48        48      48         48            48
#  Proanthocyanins Color Hue Dilution Proline cluster
#1              59    59  59       59      59      59
#2               3     3   3        3       3       3
#3               3     3   3        3       3       3
#4              65    65  65       65      65      65
#5              48    48  48       48      48      48


km.tb <- table(wine$Type, wine$cluster)
km.tb
#   1  2  3
#1 59  0  0
#2  3  3 65
#3  0 48  0


# wine 데이터의 반응변수(Type)와 군집 결과의 일치도(Agreement)를 나타내는 수정된 순위지수
# (Adjusted rank index) 구하기
library(flexclust)
randIndex(km.tb)
#      ARI
# 0.897495


