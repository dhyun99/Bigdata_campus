
# iris 데이터를 훈련용 데이터(80%)와 테스트 데이터(20%)로 분리하기
data("iris")
library(caret)
parts <- createDataPartition(iris$Species, p = 0.8)
iris.train <- iris[parts$Resample1, ]
table(iris.train$Species)
#setosa versicolor virginica
#    40         40        40

iris.test <- iris[-parts$Resample1, ]
table(iris.test$Species)
#setosa versicolor  virginica 
#    10         10         10 


# 정규혼합분포 모델 생성하기
library(mclust)
mc.fit <- Mclust(iris.train[, 1:4], G = 3)


# 적합된 모델의 추가적인 정보 확인하기
ls(mc.fit)
# [1] "bic" "BIC" "call" "classification"
# [5] "d" "data" "df" "G"
# [9] "hypvol" "loglik" "modelName" "n"
#[13] "parameters" "uncertainty" "z"

mc.fit$classification
#2   3   4   6   7   9  10  11  12  13  14  15  16  18  19  21  22  23  24  26  27  28  29  30 
#1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
#31  32  33  35  36  37  38  40  41  42  43  46  47  48  49  50  52  53  54  56  57  58  59  60 
# 1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   2   2   2   2   2   2   2 
#61  62  63  64  66  67  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  85  86  87 
# 2   2   2   2   2   2   3   2   3   2   3   2   2   2   2   3   2   2   2   2   2   2   2   2 
#88  91  92  94  95  98  99 100 101 102 103 104 105 106 107 109 111 112 113 114 115 118 119 120 
# 3   2   2   2   2   2   2   2   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
#122 124 125 126 127 129 130 131 132 133 134 137 138 140 141 142 143 144 145 146 147 148 149 150 
#  3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 


# 적합된 결과 확인하기
summary(mc.fit, parameters = TRUE)
#---------------------------------------------------- 
#  Gaussian finite mixture model fitted by EM algorithm 
#---------------------------------------------------- 
#  
#  Mclust VEV (ellipsoidal, equal shape) model with 3 components: 
#  
# log.likelihood   n df       BIC       ICL
#      -140.5116 120 38 -462.9479 -467.5478
#
#Clustering table:
# 1  2  3 
#40 35 45 
#
#Mixing probabilities:
#        1         2         3 
#0.3333333 0.2970706 0.3695961 
#
#Means:
#               [,1]     [,2]     [,3]
#Sepal.Length 4.9975 5.904657 6.507284
#Sepal.Width  3.3825 2.773096 2.925342
#Petal.Length 1.4475 4.212958 5.429678
#Petal.Width  0.2350 1.305630 1.983162
#
#Variances:
#  [,,1]
#             Sepal.Length Sepal.Width Petal.Length Petal.Width
#Sepal.Length   0.13788708 0.106096288  0.021593827 0.010767318
#Sepal.Width    0.10609629 0.148004850  0.012686864 0.007827336
#Petal.Length   0.02159383 0.012686864  0.025719490 0.005402892
#Petal.Width    0.01076732 0.007827336  0.005402892 0.008364090
#[,,2]
#             Sepal.Length Sepal.Width Petal.Length Petal.Width
#Sepal.Length   0.22491462  0.08247108   0.14788281  0.04372401
#Sepal.Width    0.08247108  0.08364643   0.08600908  0.04101474
#Petal.Length   0.14788281  0.08600908   0.18049135  0.05651135
#Petal.Width    0.04372401  0.04101474   0.05651135  0.03585869
#[,,3]
#             Sepal.Length Sepal.Width Petal.Length Petal.Width
#Sepal.Length   0.44180557   0.1324736    0.3499644  0.05722376
#Sepal.Width    0.13247356   0.1426699    0.1364356  0.07255690
#Petal.Length   0.34996437   0.1364356    0.3954817  0.10683558
#Petal.Width    0.05722376   0.0725569    0.1068356  0.09097113


# 적합된 결과 시각화하기
plot.Mclust(mc.fit)
# Model-based clustering plots: 
#  
#1: BIC
#2: classification
#3: uncertainty
#4: density
#
#선택: 2


# 테스트 데이터로 모델 성능 평가하기
mc.pred <- predict(mc.fit, newdata = iris.test[, 1:4])
mc.tb <- table(mc.pred$classification, iris.test$Species)
mc.tb
# setosa versicolor virginica
#1    10          0         0
#2     0         10         2
#3     0          0         8
mean(as.integer(iris.test$Species) == mc.pred$classification) # Accuracy
#[1] 0.9333333

(1-sum(diag(mc.tb))/sum(mc.tb))                               # Error Rate
#[1] 0.06666667


