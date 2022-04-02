
# mixtools 패키지와 faithful 데이터 읽어오기
library(mixtools)
data("faithful")
str(faithful)
#'data.frame':	272 obs. of  2 variables:
#$ eruptions: num  3.6 1.8 3.33 2.28 4.53 ...
#$ waiting  : num  79 54 74 62 85 55 88 85 51 85 ...


head(faithful)
#  eruptions waiting
#1     3.600      79
#2     1.800      54
#3     3.333      74
#4     2.283      62
#5     4.533      85
#6     2.883      55


# faithful 자료의 다음 분출까지의 대기 시간(분)에 대한 히스토그램 그리기
hist(faithful$waiting, main = "Time between Old Faithful eruptions",
     xlab = "Minutes", ylab = "Frequency",
     cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.4)


# EM 알고리즘을 이용한 정규혼합분포의 모델 생성 및 추정 결과 확인하기 
em.fit <- normalmixEM(faithful$waiting, lambda = .5, mu = c(55, 80), sigma = 5)
summary(em.fit)
#summary of normalmixEM object:
#         comp 1   comp 2
#lambda  0.36085  0.63915
#mu     54.61364 80.09031
#sigma   5.86909  5.86909
#loglik at estimate:  -1034.002 


# 적합된 모델의 추가적인 정보 확인하기
ls(em.fit)
#[1] "all.loglik" "ft" "lambda" "loglik" "mu"
#[6] "posterior" "restarts" "sigma" "x"

em.fit$all.loglik
# [1] -1051.090 -1034.168 -1034.013 -1034.003 -1034.002 -1034.002 -1034.002 -1034.002 -1034.002
#[10] -1034.002


# 추정된 정규혼합분표 시각화하기
par(mfrow = c(1, 2))
plot(em.fit, density = TRUE, cex.axis = 1.4, cex.lab = 1.4, cex.main = 1.8,
     main2 = "Time between Old Faithful eruptions", xlab2 = "Minutes")
par(mfrow = c(1, 1))

