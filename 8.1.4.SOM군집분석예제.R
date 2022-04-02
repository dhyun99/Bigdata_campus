
# rattle 패키지의 wine 데이터 읽어오기
data(wine, package = "rattle")
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
# 1  2  3 
#59 71 48 


# wine 데이터를 훈련용 데이터(80%)와 테스트 데이터(20%)로 분리하기
library(caret)
parts <- createDataPartition(wine$Type, p = 0.8)
wine.train <- wine[parts$Resample1, ]
table(wine.train$Type)
# 1  2  3
#48 57 39

wine.test <- wine[-parts$Resample1, ]
table(wine.test$Type)
# 1  2  3 
#11 14  9 


# 훈련용 데이터로 3개 군집의 SOM 군집분석 모델 생성하기
library(kohonen)
set.seed(123)
som.fit <- som(scale(wine.train[-1]),
              grid = somgrid(3, 1, "hexagonal"),
              rlen =100,
              alpha = c(0.05, 0.01))
summary(som.fit)
#SOM of size 3x1 with a hexagonal topology and a bubble neighbourhood function.
#The number of data layers is 1.
#Distance measure(s) used: sumofsquares.
#Training data included: 144 objects.
#Mean distance to the closest unit in the map: 7.04.


# 군집분석 결과 시각화하기
plot(som.fit, type = "codes")


# 적합된 모델 추가적인 정보 확인하기
ls(som.fit)
# [1] "alpha"            "changes"          "codes"            "data"             "dist.fcts"       
# [6] "distance.weights" "distances"        "grid"             "maxNA.fraction"   "radius"          
#[11] "unit.classif"     "user.weights"     "whatmap"  

som.fit$unit.classif
# [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
#[50] 3 2 3 3 3 3 3 3 3 1 3 3 3 3 3 3 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#[99] 3 3 3 3 3 3 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2


# 데이터가 학습되는 동안 유사도의 변화 그래프(왼쪽 100번, 오른쪽 500번 학습한 결과)
som.fit1 <- som(scale(wine.train[-1]), grid = somgrid(3, 1, "hexagonal"),
                rlen = 100, alpha = c(0.05, 0.01))
som.fit2 <- som(scale(wine.train[-1]), grid = somgrid(3, 1, "hexagonal"),
                rlen = 500, alpha = c(0.05, 0.01))
par(mfrow = c(1, 2))
plot(som.fit1, type = "changes", main = "Wine data: SOM(Learning no = 100)")
plot(som.fit2, type = "changes", main = "Wine data: SOM(Learning no = 500)")
par(mfrow = c(1, 1))


# 테스트 데이터로 모델의 예측된 분류 생성하기
pred.som <- predict(som.fit, newdata = scale(wine.test[-1]))


#예측 군집 결과의 승자 유니트 값을 테스트 데이터 속성(cluster)에 추가하기
wine.test$cluster <- pred.som$unit.classif
aggregate(wine.test,
          by = list(cluster = wine.test$cluster, Type = wine.test$Type), NROW)
#  cluster Type Type Alcohol Malic Ash Alcalinity Magnesium Phenols Flavanoids Nonflavanoids
#1       1    1   11      11    11  11         11        11      11         11            11
#2       1    2    2       2     2   2          2         2       2          2             2
#3       2    2    1       1     1   1          1         1       1          1             1
#4       3    2   11      11    11  11         11        11      11         11            11
#5       2    3    9       9     9   9          9         9       9          9             9
#  Proanthocyanins Color Hue Dilution Proline cluster
#1              11    11  11       11      11      11
#2               2     2   2        2       2       2
#3               1     1   1        1       1       1
#4              11    11  11       11      11      11
#5               9     9   9        9       9       9


# 모델 성능 평가하기
wine.test$Type2 <- ifelse(wine.test$Type == "1", 1,
                      ifelse(wine.test$Type == "2", 3, 2))
som.tb <- table(wine.test$Type2, wine.test$cluster)
som.tb
#   1  2  3
#1 10  0  1
#2  0  9  0
#3  1  0 13

mean(wine.test$Type2 == wine.test$cluster)
#[1] 0.9411765

(1-sum(diag(som.tb))/sum(som.tb))
#[1] 0.05882353

