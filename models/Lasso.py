from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools
import numpy as np
from matplotlib import pyplot as plt, transforms
import random
from helper import moving_filter, mse, mae, dtw

class Model():
    # Function to extract features
    def GetFeatures(self, intras, extras, simple, target):
        X = []
        Y = []
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


            featureList = []
            
            if(simple):
                featureList = [d2eAP, deAP, eAP, t]
            else:
                featureList = [d2eAP, deAP, eAP, np.exp(d2eAP), np.exp(deAP), np.exp(eAP), t]
        

    #         maxInd = np.argmax(eAP[:1500])
    #         minInd = 1500 + np.argmin(eAP[1500:])

    #         distFromMax = [i for i in range(len(extras[0]))]
    #         distFromMin = [i for i in range(len(extras[0]))]

    #         distFromMax = np.exp(-np.absolute(distFromMax - maxInd))
    #         distFromMin = np.exp(-np.absolute((distFromMin - minInd)/50))

    #         featureList.append(distFromMax)
    #         featureList.append(distFromMin)
                
            featureMatrix = []
            
            for comb in combs[1:]:
                for tpl in comb:
                    feature = 1
                    
                    for ind in tpl:
                        feature = feature*featureList[ind]
                        
                    featureMatrix.append(feature)
            
            
    ######### Uncomment the following to add a parameter for some finite timesteps
    ######### around positive peak in eAP and local minima in latter half of eAP
    #         oneHotMatrix = []
    #         maxInd = np.argmax(eAP[:1500])
    #         minInd = 1500 + np.argmin(eAP[1500:])


    #         for j in range(maxInd-50, maxInd+50):
    #             temp = list(np.zeros(len(eAP)))
    #             temp[j] = 1
    #             oneHotMatrix.append(temp)

    #         for j in range(100):
    #             temp = list(np.zeros(len(eAP)))

    #             for i in range(minInd-500+10*j, minInd-500+10*j+10):
    #                 temp[i] = 1

    #             oneHotMatrix.append(temp)

    #         featureMatrix.extend(oneHotMatrix)
            
            X.extend(np.array(featureMatrix).T)

            if(target == 0):
                Y.extend(iAP)
            elif(target == 1):
                Y.extend(diAP)
            else:
                Y.extend(d2iAP)
            
        X = np.array(X)
        Y = np.array(Y)
        
        return X, Y


    def RunModel(self, intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest, simple, target):
        print("Extracting features ...")
        # Extract features from training data
        X0, Y0 = self.GetFeatures(intrasTrain, extrasTrain, simple, target)
        scaler0 = StandardScaler()
        X0 = scaler0.fit_transform(X0)

        # Extract features from validation and test data for hyperparam tuning and evaluation
        X_val0, Y_val0 = self.GetFeatures(intrasVal, extrasVal, simple, target)
        X_val0 = scaler0.transform(X_val0)

        X_test0, Y_test0 = self.GetFeatures(intrasTest, extrasTest, simple, target)
        X_test0 = scaler0.transform(X_test0)

        print("Training the model ...")
        # Linear Regression model using Scikit and Lasso selection
        lamda = 0.0
        
        if(simple):
            if(target == 0):
                lamda = 0.00001
            elif(target == 1):
                lamda = 0.00001
            else:
                lamda = 0.000001
        else:
            if(target == 0):
                lamda = 0.01
            elif(target == 1):
                lamda = 0.00001
            else:
                lamda = 0.0000005
        
        selector = SelectFromModel(Lasso(alpha=lamda, random_state=10))
        X = X0
        Y = Y0
        X_val = X_val0
        Y_val = Y_val0
        X_test = X_test0
        Y_test = Y_test0
        selector.fit(X, Y)

#         print(selector.get_support())

        X_selected = selector.transform(X)
        reg_selected = Lasso(alpha=lamda, random_state=10).fit(X_selected, Y)

#         print(reg_selected.coef_)
#         print(reg_selected.intercept_)

#         predTrain = reg_selected.predict(X_selected)
#         print("MSE on train with reg_selected", mse(predTrain, Y))
#         print("MAE on train with reg_selected", mae(predTrain, Y))


        # Evaluate model on val data
        out = np.array(reg_selected.predict(selector.transform(X_val))).reshape((-1, 8000))
        prediAP_selected = []

        if(target == 2):
            prediAP_selected = np.cumsum(np.cumsum(out, axis=1), axis=1)
        elif(target == 1):
            prediAP_selected = np.cumsum(out, axis=1)
        else:
            prediAP_selected = out


        print("MSE on val with reg_selected", mse(prediAP_selected, intrasVal))
        print("MAE on val with reg_selected", mae(prediAP_selected, intrasVal))
        print("DTW on val with reg_selected", dtw(prediAP_selected, intrasVal))


        # Evaluate model on test data
        out = np.array(reg_selected.predict(selector.transform(X_test))).reshape((-1, 8000))
        prediAP_selected = []

        if(target == 2):
            prediAP_selected = np.cumsum(np.cumsum(out, axis=1), axis=1)
        elif(target == 1):
            prediAP_selected = np.cumsum(out, axis=1)
        else:
            prediAP_selected = out


        print("MSE on test with reg_selected", mse(prediAP_selected, intrasTest))
        print("MAE on test with reg_selected", mae(prediAP_selected, intrasTest))
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

        print(np.array(featureCombs)[selector.get_support()])


        # Choose a recording at random from the test set
        ind = random.randint(0, len(extrasTest)-1)
        print(ind)


        # Prediction plots
        transparency = 0.7
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        fig0.tight_layout(pad=5)

        ax0.plot(prediAP_selected[ind], linestyle='dashed')
        ax0.plot(intrasTest[ind], alpha=transparency, color='red')
        ax0.legend(['Prediction', 'Ground Truth'])#, prop={'size': 18})
        ax0.title.set_text("Intra-cellular AP")
        ax0.title.set_fontsize(20)
        ax0.set_xlabel("Timesteps", fontsize=20)
        ax0.set_ylabel("iAP", fontsize=20)
        # ax0.set_ylim([-0.2, 1.1])
        # ax0.tick_params(axis='x', labelsize=16)
        # ax0.tick_params(axis='y', labelsize=16)