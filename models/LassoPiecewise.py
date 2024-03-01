from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
import numpy as np
from matplotlib import pyplot as plt, transforms
import random
from helper import moving_filter, mse, mae, dtw

class Model():
    def GetFeatures(self, intras, extras, simple, target):
        XBefore = []
        XAfter = []
        YBefore = []
        YAfter = []
        indicesBefore = []
        indicesAfter = []
        windowSize = 20
        featureNames = []
        
        if(simple):
            featureNames = ['d2eAP', 'deAP', 'eAP', 't']
        else:
            featureNames = ['d2eAP', 'deAP', 'eAP', 'exp(d2eAP)', 'exp(deAP)', 'exp(eAP)', 't']

        s = [i for i in range(len(featureNames))]
        combs = [list(itertools.combinations(s, r)) for r in range(len(s)+1)]
        featureCombs = []
        
        for comb in combs[1:]:
            for tpl in comb:
                name = ""

                for ind in tpl:
                    name += "*"+featureNames[ind]

                featureCombs.append(name)
                
        t = np.array([i for i in range(len(extras[0]))])/len(extras[0])
        
        for i in range(len(extras)):
            iAP = intras[i]
            diAP = np.array(moving_filter(np.gradient(iAP), windowSize))
            d2iAP = np.array(moving_filter(np.gradient(diAP), windowSize))

            eAP = extras[i]
            deAP = np.array(moving_filter(np.gradient(eAP), windowSize))
            d2eAP = np.array(moving_filter(np.gradient(deAP), windowSize))
            
            breakInd = np.argmax(eAP[:1500])

            featureList1 = []
            featureList2 = []

            if(simple):
                featureList1 = [d2eAP[:breakInd], deAP[:breakInd], eAP[:breakInd], t[:breakInd]]
                featureList2 = [d2eAP[breakInd:], deAP[breakInd:], eAP[breakInd:], t[breakInd:]]
            else:
                featureList1 = [d2eAP[:breakInd], deAP[:breakInd], eAP[:breakInd], t[:breakInd], np.exp(d2eAP)[:breakInd], 
                                np.exp(deAP)[:breakInd], np.exp(eAP)[:breakInd], t[:breakInd]]
                featureList2 = [d2eAP[breakInd:], deAP[breakInd:], eAP[breakInd:], t[breakInd:], np.exp(d2eAP)[breakInd:], 
                                np.exp(deAP)[breakInd:], np.exp(eAP)[breakInd:], t[breakInd:]]

            featureMatrix1 = []
            featureMatrix2 = []
            
            for comb in combs[1:]:
                for tpl in comb:
                    feature1 = 1
                    feature2 = 1
                    
                    for ind in tpl:
                        feature1 = feature1*featureList1[ind]
                        feature2 = feature2*featureList2[ind]
                        
                    featureMatrix1.append(feature1)
                    featureMatrix2.append(feature2)
            
            start = int(len(XBefore))
            XBefore.extend(np.array(featureMatrix1).T)
            end = int(len(XBefore))-1
            indicesBefore.append([start, end])
            
            start = int(len(XAfter))
            XAfter.extend(np.array(featureMatrix2).T)
            end = int(len(XAfter))-1
            indicesAfter.append([start, end])
            
            if(target == 0):
                YBefore.extend(iAP[:breakInd])
                YAfter.extend(iAP[breakInd:])
            elif(target == 1):
                YBefore.extend(diAP[:breakInd])
                YAfter.extend(diAP[breakInd:])
            else:
                YBefore.extend(d2iAP[:breakInd])
                YAfter.extend(d2iAP[breakInd:])
            
        XBefore = np.array(XBefore)
        XAfter = np.array(XAfter)
        YBefore = np.array(YBefore)
        YAfter = np.array(YAfter)
        
        return XBefore, XAfter, YBefore, YAfter, indicesBefore, indicesAfter


    def RunModel(self, intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest, simple, target):
        print("Extracting features ...")
        # Extract features from training data
        XBefore, XAfter, YBefore, YAfter, indicesBefore, indicesAfter = self.GetFeatures(intrasTrain, extrasTrain, simple, target)
        scalerBefore = StandardScaler()
        scalerAfter = StandardScaler()

        XBefore = scalerBefore.fit_transform(XBefore)
        XAfter = scalerAfter.fit_transform(XAfter)

        # Extract features from validation and test data for hyperparam tuning and evaluation
        X_valBefore, X_valAfter, Y_valBefore, Y_valAfter, indicesBefore_val, indicesAfter_val = self.GetFeatures(intrasVal, extrasVal, simple, target)
        X_valBefore = scalerBefore.transform(X_valBefore)
        X_valAfter = scalerAfter.transform(X_valAfter)

        X_testBefore, X_testAfter, Y_testBefore, Y_testAfter, indicesBefore_test, indicesAfter_test = self.GetFeatures(intrasTest, extrasTest, simple, target)
        X_testBefore = scalerBefore.transform(X_testBefore)
        X_testAfter = scalerAfter.transform(X_testAfter)

        print("Training the model ...")
        # Linear Regression model using Scikit and Lasso selection
        lamda1 = 0.01
        lamda2 = 0.01
        
        if(simple):
            if(target == 0):
                lamda1 = 0.01
                lamda2 = 0.01
            elif(target == 1):
                lamda1 = 0.00001
                lamda2 = 0.00001
            else:
                lamda1 = 0.0000001
                lamda2 = 0.0000001
        else:
            if(target == 0):
                lamda1 = 0.01
                lamda2 = 0.01
            elif(target == 1):
                lamda1 = 0.00001
                lamda2 = 0.00001
            else:
                lamda1 = 0.00001
                lamda2 = 0.0000001

        selector1 = SelectFromModel(Lasso(alpha=lamda1, random_state=10))
        selector2 = SelectFromModel(Lasso(alpha=lamda2, random_state=10))

        selector1.fit(XBefore, YBefore)
        selector2.fit(XAfter, YAfter)

#         print(selector1.get_support())
#         print(selector2.get_support())

        XBefore_selected = selector1.transform(XBefore)
        XAfter_selected = selector2.transform(XAfter)

        reg_selected1 = Lasso(alpha=lamda1, random_state=10).fit(XBefore_selected, YBefore)
        reg_selected2 = Lasso(alpha=lamda2, random_state=10).fit(XAfter_selected, YAfter)

#         print(reg_selected1.coef_)
#         print(reg_selected1.intercept_)

#         print(reg_selected2.coef_)
#         print(reg_selected2.intercept_)


        # Evaluate model on validation data
        valOut = []
        for i in range(len(indicesBefore_val)):
            valOut1 = reg_selected1.predict(selector1.transform(X_valBefore[indicesBefore_val[i][0]:indicesBefore_val[i][1]+1])).reshape((-1))
            valOut2 = reg_selected2.predict(selector2.transform(X_valAfter[indicesAfter_val[i][0]:indicesAfter_val[i][1]+1])).reshape((-1))
            valOut.extend(np.concatenate([valOut1, valOut2], 0))
            
        valOut = np.reshape(np.array(valOut), (-1, 8000))

        prediAP_selected = []

        if(target == 2):
            prediAP_selected = np.cumsum(np.cumsum(valOut, axis=1), axis=1)
        elif(target == 1):
            prediAP_selected = np.cumsum(valOut, axis=1)
        else:
            prediAP_selected = valOut

        print("MSE on val with reg_selected", mse(prediAP_selected, intrasVal))
        print("MAE on val with reg_selected", mae(prediAP_selected, intrasVal))
        print("DTW on val with reg_selected", dtw(prediAP_selected, intrasVal))



        # Evaluate model on test data
        testOut = []
        for i in range(len(indicesBefore_test)):
            testOut1 = reg_selected1.predict(selector1.transform(X_testBefore[indicesBefore_test[i][0]:indicesBefore_test[i][1]+1])).reshape((-1))
            testOut2 = reg_selected2.predict(selector2.transform(X_testAfter[indicesAfter_test[i][0]:indicesAfter_test[i][1]+1])).reshape((-1))
            testOut.extend(np.concatenate([testOut1, testOut2], 0))
            
        testOut = np.reshape(np.array(testOut), (-1, 8000))

        if(target == 2):
            prediAP_selected = np.cumsum(np.cumsum(testOut, axis=1), axis=1)
        elif(target == 1):
            prediAP_selected = np.cumsum(testOut, axis=1)
        else:
            prediAP_selected = testOut


        print("MSE on test with reg_selected", mse(prediAP_selected, intrasTest))
        print("MAE on test with regselected", mae(prediAP_selected, intrasTest))
        print("DTW on test with reg_selected", dtw(prediAP_selected, intrasTest))


        # Important feature selection using Lasso
        featureNames = []
        if(simple):
            featureNames = ['d2eAP', 'deAP', 'eAP', 't']
        else:
            featureNames = ['d2eAP', 'deAP', 'eAP', 'np.exp(d2eAP)', 'np.exp(deAP)', 'np.exp(eAP)', 't']
            
        s = [i for i in range(len(featureNames))]
        combs = [list(itertools.combinations(s, r)) for r in range(len(s)+1)]
        featureCombs = []

        for comb in combs[1:]:
            for tpl in comb:
                name = ""

                for ind in tpl:
                    name += "*"+featureNames[ind]

                featureCombs.append(name)

        #### Only for weight params
        # featureCombs.extend(["max"+str(i) for i in range(-50, 50)])
        # featureCombs.extend(["min"+str(i) for i in range(100)])
        ####

        print(np.array(featureCombs)[selector1.get_support()])
        print(np.array(featureCombs)[selector2.get_support()])


        # Choose a recording at random from the test set
        ind = random.randint(0, len(extrasTest)-1)#219, 133, 48
        print(ind)


        # Prediction plots
        transparency = 0.7
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        fig0.tight_layout(pad=4)

        # Plot first derivative
        ax0.plot(prediAP_selected[ind], linestyle='dashed')
        ax0.plot(intrasTest[ind], alpha=transparency, color='red')
        ax0.legend(['Prediction', 'Ground Truth'], prop={'size': 18})
        ax0.title.set_text("Intra-cellular AP")
        ax0.title.set_fontsize(20)
        ax0.set_xlabel("Timesteps", fontsize=20)
        ax0.set_ylabel("iAP", fontsize=20)
        # ax0.set_ylim([-0.1, 1.1])
        # ax0.tick_params(axis='x', labelsize=16)
        # ax0.tick_params(axis='y', labelsize=16)