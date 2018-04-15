#机器学习包-mrl的使用
#集成学习比较/二分类问题
#要求：数据集X都是数值型，X没有因子型; 输出的y是概率型，而不是类别
#进行前：one-hot处理；平衡样本
#建模逻辑：先尝试多个分类模型，对有希望的模型进行调参数

#试验数据：spam【邮件垃圾分类数据】

library(mlr)

setwd("/Users/apple/code_tool/R/datamining")

#?measures  #查看评价指标，根据任务确定：返回值、任务类型

#--------------------------------------1定义学习任务
#--1.1数据来源
data(spam, package = 'kernlab')
classif.task = makeClassifTask(id = "spam", data = spam, target = "type")
classif.task
#--1.2默认包定义好的任务名（二者等价）
#getTaskData(spam.task)
#二分类；只有数值型变量
str(spam)

#--------------------------------------2多个分类模型比较
#---选择可用模型：会根据任务给出模型建议（设定要求：输出是概率）
lrns_t = listLearners(classif.task, properties = "prob")

#--基本分类模型+集成模型
#记得指定预测类型p与学习器id
lrns = list(makeLearner("classif.ksvm", predict.type = "prob", id = "svm"), 
            makeLearner("classif.rpart", predict.type = "prob", id = "rpart"), #可NA
            #makeLearner("classif.naiveBayes", predict.type = "prob", id = "naiveBayes"), #可NA
            makeLearner("classif.kknn", predict.type = "prob", id = "kknn"),
            makeLearner("classif.nnet", predict.type = "prob", id = "nnet"),
            makeLearner("classif.logreg", predict.type = "prob", id = "logreg"),
            #以下为集成学习器
            makeLearner("classif.xgboost", predict.type = "prob", id = "xgboost"),
            makeLearner("classif.randomForest", predict.type = "prob", id = "rf"),
            makeLearner("classif.blackboost", predict.type = "prob", id = "GBRT"),#耗时：放弃调试
            #makeLearner("classif.boosting", predict.type = "prob", id = "AdaBoost"), #特别耗时
            makeLearner("classif.gbm", predict.type = "prob", id = "GBM"))

rdesc = makeResampleDesc("CV", iters = 10)  #10折交叉验证
meas = list(logloss, auc, ssr,timetrain)   #概率p的评价指标
bmr = benchmark(lrns, classif.task,         #多个模型进行运算
                resamplings = rdesc, 
                measures = meas,  
                show.info = FALSE)
bmr
#--将比较结果存为数据框形式
perf = getBMRPerformances(bmr, as.df = TRUE)   
head(perf)

#--------2比较结果可视化
#--小提琴图
library(ggplot2)
plotBMRBoxplots(bmr, measure = auc, style = "violin", pretty.names = FALSE) +
  aes(color = learner.id) +
  theme(strip.text.x = element_text(size = 8))
#--3个指标评价排名
cbind(convertBMRToRankMatrix(bmr, logloss),
      convertBMRToRankMatrix(bmr, auc),
      convertBMRToRankMatrix(bmr, ssr))

#--------------------------------------3单个模型调参
#---------------3.1xgboost
##结论：进行调参后，xgboost还是大有作为的
#得到默认参数设置
getParamSet("classif.xgboost")
#--先设置不可调部分：评价指标
classif.xgboost = makeLearner("classif.xgboost", predict.type = "prob", 
                              id = "xgboost",
                              eval_metric="logloss")  #--指定优化指标
#--对超参数设置搜索范围
ps = makeParamSet(
  #控制模型复杂度(树深、最小节点实例数、划分最小损失减少量)
  makeIntegerParam("max_depth", lower = 0, upper = 50),
  makeNumericParam("min_child_weight", lower = 0, upper = 6, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -1, upper = 6, trafo = function(x) 2^x),
  makeNumericParam("lambda", lower = -10, upper = 10, trafo = function(x) 2^x),
  #增加随机性，使模型更稳健
  makeNumericParam("subsample", lower = 0, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.3, upper = 1),
  makeNumericParam("eta", lower = -3, upper = 1, trafo = function(x) 10^x),
  makeIntegerParam("nrounds", lower = 1, upper = 100),
  #类平衡设置
  makeNumericParam("max_delta_step", lower = 0, upper = 16)
)

ctrl = makeTuneControlRandom(maxit = 200L)  #200次随机搜索
rdesc = makeResampleDesc("Holdout")         #重采样
res = tuneParams(classif.xgboost, classif.task, 
                 resampling = rdesc, 
                 measures = meas,  #如果该模型可以优化指标，那么会默认优化第一个指标
                 par.set = ps, control = ctrl, 
                 show.info = FALSE)
res      #最优结果
#查看所有参数结果
#法一：(存疑)
#generateHyperParsEffectData(res)
#法二：不必指定measures
#ll=as.data.frame(res$opt.path)   #有很多因为错误没有运行
#sum(is.na(ll$exec.time))  #无效运行

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
xgboost2 = setHyperPars(classif.xgboost, par.vals = res$x)
xgboost2
xgboost2 = setLearnerId(xgboost2, "xgboost2")  #重命名学习器


#------用一开始的数据集运行最优结果
#提取第一次的重采样结果用于随后各个模型，保证数据集划分完全一致
rin = getBMRPredictions(bmr)[[1]][[1]]$instance
rin      #第一次的重采样结果
bmr2 = benchmark(xgboost2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2












#-----------------------------------------4合并模型结果并进行可视化展示
bmr_all = mergeBenchmarkResults(list(bmr, bmr2))  
bmr_all
#--可视化
#--小提琴图
library(ggplot2)
plotBMRBoxplots(bmr_all, measure = auc, style = "violin", pretty.names = FALSE) +
  aes(color = learner.id) +
  theme(strip.text.x = element_text(size = 8))
#--3个指标评价排名
cbind(convertBMRToRankMatrix(bmr_all, logloss),
      convertBMRToRankMatrix(bmr_all, auc),
      convertBMRToRankMatrix(bmr_all, ssr))


#-----结论：
#集成算法会使模型有较大的提示，但单个算法几乎很难提升太多
#但集成算法对参数的调整要求比较高
#从箱线图可以看出，波动较大的，调节的价值较高






#-----------------------3.2svm调参
#--结论：搜索调参后效果反而变差了
#--classif.ksvm比classif.svm运行速度要快很多
getParamSet("classif.ksvm")
classif.ksvm = makeLearner("classif.ksvm", predict.type = "prob", id = "svm")
#--搜索范围
ps = makeParamSet(
  makeNumericParam("C", lower = -10, upper = 10, trafo = function(x) 2^x),
  makeDiscreteParam("kernel", values = c("vanilladot", "polydot", "rbfdot")),
  makeNumericParam("sigma", lower = -10, upper = 10, trafo = function(x) 2^x,
                   requires = quote(kernel == "rbfdot")),
  makeIntegerParam("degree", lower = 2L, upper = 5L,
                   requires = quote(kernel == "polydot"))
)
#对于有些不能使用随机搜索，要采用限制最大试验数的方法 #ctrl = makeTuneControlRandom(maxit = 100L) 
ctrl = makeTuneControlIrace(maxExperiments = 200L)
rdesc = makeResampleDesc("Holdout")
res = tuneParams(classif.ksvm, iris.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)
res
#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
svm2 = setHyperPars(classif.ksvm, par.vals = res$x)
svm2
svm2 = setLearnerId(svm2, "svm2")  #重命名学习器
#--用于一开始的数据集
bmr2 = benchmark(svm2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2


#-------------- ---------3.3rf
#rf调参与否影响不大，因为大部分时候有它自己的默认规则
#注意：相对xgboost速度明显下降
#--得到默认参数设置
getParamSet("classif.randomForest")
classif.randomForest = makeLearner("classif.randomForest",
                                   predict.type = "prob", id = "rf")
#--对超参数设置搜索范围
ps = makeParamSet(
  makeIntegerParam("ntree", lower = 4, upper = 9, trafo = function(x) 2^x),
  #makeIntegerParam("mtry", lower = 6, upper = 50),
  makeIntegerParam("nodesize", lower = 1, upper = 30)
)
ctrl = makeTuneControlRandom(maxit = 100L)  #200次随机搜索
rdesc = makeResampleDesc("Holdout")         #重采样
res = tuneParams(classif.randomForest, classif.task, rdesc, 
                 par.set = ps, control = ctrl, 
                 show.info = FALSE)
res      #最优结果
#查看所有参数结果
#ll=as.data.frame(res$opt.path)   #有很多因为错误没有运行
#sum(is.na(ll$exec.time))  #无效运行

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
randomForest2 = setHyperPars(classif.randomForest, par.vals = res$x)
randomForest2
randomForest2 = setLearnerId(randomForest2, "randomForest2")  #重命名学习器
#--用于一开始的数据集
bmr2 = benchmark(randomForest2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2

#--------根据最优结果具体建模rf: res
library(randomForest)
set.seed(1234)
fit.forest = randomForest(type~., data=spam,
                          importance=T)
varImpPlot(fit.forest)
importance(fit.forest)



#---------------3.4glmboost
#--得到默认参数设置
getParamSet("classif.glmboost")
#--参数范围
ps = makeParamSet(
  makeIntegerParam("mstop", lower = 0, upper = 9, trafo = function(x) 2^x),
  makeDiscreteParam("risk", values = c("inbag", "oobag", "none")),
  makeNumericParam("nu", lower = 0, upper = 1)
)
glmboost = makeLearner("classif.glmboost", predict.type = "prob", id = "glmboost")
ctrl = makeTuneControlIrace(maxExperiments = 100L)
rdesc = makeResampleDesc("Holdout")
res = tuneParams(glmboost, classif.task, rdesc, par.set = ps, control = ctrl, show.info = FALSE)
res
#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
glmboost2 = setHyperPars(glmboost, par.vals = res$x)
glmboost2
glmboost2 = setLearnerId(glmboost2, "glmboost2")  #重命名学习器
#--用于一开始的数据集
bmr2 = benchmark(glmboost2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2



#-------------------------------------3.5rpart调试
#--可以设置损失矩阵
#得到默认参数设置
getParamSet("classif.rpart")

classif.rpart = makeLearner("classif.rpart", predict.type = "prob", 
                            id = "rpart")

#--对超参数设置搜索范围  #?rpart.control 
ps = makeParamSet(
  #控制模型复杂度(最小划分节点实例数、树深、复杂度参数)
  makeIntegerParam("minsplit", lower = 1, upper = 50),
  makeIntegerParam("maxdepth", lower = 1, upper = 30),
  makeNumericParam("cp", lower = -10, upper = 0, trafo = function(x) 10^x)
)

ctrl = makeTuneControlRandom(maxit = 200L)  #200次随机搜索
rdesc = makeResampleDesc("Holdout")         #重采样
res = tuneParams(classif.rpart, classif.task, 
                 resampling = rdesc,
                 measures = meas,
                 par.set = ps, control = ctrl, 
                 show.info = FALSE)
res      #最优结果

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
rpart2 = setHyperPars(classif.rpart, par.vals = res$x)
rpart2
rpart2 = setLearnerId(rpart2, "rpart2")  #重命名学习器


#------用一开始的数据集运行最优结果
#提取第一次的重采样结果用于随后各个模型，保证数据集划分完全一致
rin = getBMRPredictions(bmr)[[1]][[1]]$instance
rin      #第一次的重采样结果
bmr2 = benchmark(rpart2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2

#--------根据最优结果具体建模rpart: res
library(rpart)
set.seed(1234)
dtree = rpart(type~., data = spam, method = 'class')
#dtree$cptable
dtree.prune = prune(dtree, minsplit=28,maxdepth=24,cp=0.000464)
library(rpart.plot)   #画图
prp(dtree.prune, 
    type = 2, extra = 104,
    fallen.leaves = T, main='Decision Tree')
#预测
pred = predict(dtree.prune, spam, type = 'prob')
library(ModelMetrics)
logLoss(spam$type,pred)




#-------------------------------3.6AdaBoost调试
#--ada是最耗时的集成
#--可以设置损失矩阵
#得到默认参数设置
getParamSet("classif.boosting")

classif.ada = makeLearner("classif.boosting", predict.type = "prob", 
                          id = "ada")

#--对超参数设置搜索范围  #基于rpart.control 
ps = makeParamSet(
  makeDiscreteParam("mfinal", values = c(10,20,50,100,150)),
  makeDiscreteParam("coeflearn", values = c("Breiman", "Freund", "Zhu")),
  #rpart.control控制模型复杂度(最小划分节点实例数、树深、复杂度参数)
  makeIntegerParam("minsplit", lower = 1, upper = 50),
  makeIntegerParam("maxdepth", lower = 1, upper = 30),
  makeNumericParam("cp", lower = -10, upper = 0, trafo = function(x) 10^x)
)

##-----------------------------------------3.7GBRT调试
#--运行调试费时间
#得到默认参数设置
getParamSet("classif.blackboost")

classif.GBRT = makeLearner("classif.blackboost", predict.type = "prob", 
                           id = "GBRT")

#--对超参数设置搜索范围  #基于rpart.control 
ps = makeParamSet(
  #makeDiscreteParam("family", values = c("Binomial", "AdaExp", "AUC",'custom.family')),
  #--boost控制参数
  makeNumericParam("nu", lower = -10, upper = 0, trafo = function(x) 10^x),
  makeDiscreteParam("risk", values = c("inbag", "oobag", "none")),
  makeIntegerParam("mstop", lower = 4, upper = 9, trafo = function(x) 2^x),
  #tree.control控制模型复杂度(最小划分节点实例数、树深、复杂度参数)
  #makeIntegerParam("minsplit", lower = 1, upper = 50),
  #makeIntegerParam("mtry", lower = 1, upper = 50),  #与变量个数有关
  makeIntegerParam("maxdepth", lower = 1, upper = 30)
)

ctrl = makeTuneControlRandom(maxit = 50L)  #50次随机搜索
rdesc = makeResampleDesc("Holdout")         #重采样
res = tuneParams(classif.GBRT, classif.task, 
                 resampling = rdesc,
                 measures = meas,
                 par.set = ps, control = ctrl, 
                 show.info = FALSE)
res      #最优结果
best_par_vals_GBRT = res$x  #保留最优的参数结果

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
GBRT2 = setHyperPars(classif.GBRT, par.vals = res$x)
GBRT2
GBRT2 = setLearnerId(GBRT2, "GBRT2")  #重命名学习器


#------用一开始的数据集运行最优结果
bmr2 = benchmark(GBRT2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2



#-------------------------------------------------3.8GBM调试
#得到默认参数设置
getParamSet("classif.gbm")

classif.GBM = makeLearner("classif.gbm", predict.type = "prob", 
                          id = "GBM")

#--对超参数设置搜索范围  
ps = makeParamSet(
  makeIntegerParam("n.trees", lower = 1L, upper = 100L),  
  makeNumericParam("shrinkage", lower = -5, upper = 0, trafo = function(x) 10^x)
)

ctrl = makeTuneControlRandom(maxit = 100L)  #50次随机搜索
rdesc = makeResampleDesc("Holdout")         #重采样
res = tuneParams(classif.GBM, classif.task, 
                 resampling = rdesc,
                 measures = meas,
                 par.set = ps, control = ctrl, 
                 show.info = FALSE)
res      #最优结果
best_par_vals_GBM = res$x  #保留最优的参数结果

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
GBM2 = setHyperPars(classif.GBM, par.vals = res$x)
GBM2
GBM2 = setLearnerId(GBM2, "GBM2")  #重命名学习器


#------用一开始的数据集运行最优结果
bmr2 = benchmark(GBM2, classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2




