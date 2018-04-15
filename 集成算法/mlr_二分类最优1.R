#mrl
#二分类问题
#要求：数据集X都是数值型，X没有因子型; 输出的y是概率型，而不是类别
#进行前：one-hot处理；平衡样本
#建模逻辑：
#一般算法+集成算法，其中务必对集成算法进行调试
#一般算法：svm/rpart(调参)/logreg/nnet/knn
#集成算法：GBM(调参)、RF、XGBoost(调参)

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
            makeLearner("classif.kknn", predict.type = "prob", id = "kknn"),
            makeLearner("classif.nnet", predict.type = "prob", id = "nnet"),
            makeLearner("classif.logreg", predict.type = "prob", id = "logreg"),
            #以下为集成学习器
            makeLearner("classif.xgboost", predict.type = "prob", id = "xgboost"),
            makeLearner("classif.randomForest", predict.type = "prob", id = "rf"),
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


#----------------------------------------------3单个模型调参,保留最优
#xgboost、rpart、GBM
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

#--根据寻优参数结果，设置模型的超参数：par.vals = res$x
xgboost2 = setHyperPars(classif.xgboost, par.vals = res$x)
xgboost2
xgboost2 = setLearnerId(xgboost2, "xgboost2")  #重命名学习器


#-------------------------------------3.2rpart调试
#--可以设置损失矩阵
#得到默认参数设置    #getParamSet("classif.rpart")
classif.rpart = makeLearner("classif.rpart", predict.type = "prob", 
                            id = "rpart")

#--对超参数设置搜索范围  #?rpart.control 
ps = makeParamSet(
  #控制模型复杂度(最小划分节点实例数、树深、复杂度参数)
  makeIntegerParam("minsplit", lower = 1, upper = 50),
  makeIntegerParam("maxdepth", lower = 1, upper = 30),
  makeNumericParam("cp", lower = -10, upper = 0, trafo = function(x) 10^x)
)

ctrl = makeTuneControlRandom(maxit = 100L)  #200次随机搜索
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



#-------------------------------------------------3.3GBM调试
#得到默认参数设置  #getParamSet("classif.gbm")
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












#-----------------------------------------4合并模型结果并进行可视化展示
#------用一开始的数据集运行最优结果
#提取第一次的重采样结果用于随后各个模型，保证数据集划分完全一致
rin = getBMRPredictions(bmr)[[1]][[1]]$instance
rin      #第一次的重采样结果
bmr2 = benchmark(list(xgboost2, rpart2, GBM2), 
                 classif.task, 
                 rin, meas, 
                 show.info = FALSE)
bmr2
#---
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
#集成算法会使模型有较大的提升，但单个算法几乎很难提升太多
#但集成算法对参数的调整要求比较高
#从箱线图可以看出，波动较大的，调节的价值较高（不一定）
#各个学习器的学习速度也不一样
















