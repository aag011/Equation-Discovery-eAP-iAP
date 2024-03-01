import copy
import pysindy as ps
from sklearn.linear_model import Lasso
import numpy as np
from matplotlib import pyplot as plt, transforms
import random
from helper import moving_filter, mse, mae, dtw


class Model():
    # Function to extract features
    def GetFeatures(self, intras, extras, simple, target):
        XBefore = []
        XAfter = []
        YBefore = []
        YAfter = []
        windowSize = 20
                
        t = np.array([i for i in range(len(extras[0]))])/len(extras[0])
        
        for i in range(len(extras)):
            iAP = intras[i]
            diAP = np.array(moving_filter(np.gradient(iAP), windowSize))
            d2iAP = np.array(moving_filter(np.gradient(diAP), windowSize))

            eAP = extras[i]
            eAP[:1500][eAP[:1500] < np.mean(eAP)] = np.mean(eAP)
            deAP = np.array(moving_filter(np.gradient(eAP), windowSize))
            d2eAP = np.array(moving_filter(np.gradient(deAP), windowSize))
            
            breakInd = np.argmax(eAP[:1500])#+300
            
            overlap = 10
            breakInd += overlap
            featureMatrix1 = [d2eAP[:breakInd], deAP[:breakInd], eAP[:breakInd], t[:breakInd]]
            breakInd -= overlap
            featureMatrix2 = [d2eAP[breakInd:], deAP[breakInd:], eAP[breakInd:], t[breakInd:]]
            
            
            XBefore.append(np.array(featureMatrix1).T)
            XAfter.append(np.array(featureMatrix2).T)

            if(target == 0):
                YBefore.append(iAP[:breakInd+overlap])
                YAfter.append(iAP[breakInd:])
            elif(target == 1):
                YBefore.append(diAP[:breakInd+overlap])
                YAfter.append(diAP[breakInd:])
            else:
                YBefore.append(d2iAP[:breakInd+overlap])
                YAfter.append(d2iAP[breakInd:])
        
        return XBefore, XAfter, YBefore, YAfter


    def RunModel(self, intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest, simple, target):
        print("Extracting features ...")
        
        # Extract features from train, val and test data
        XBefore, XAfter, YBefore, YAfter = self.GetFeatures(intrasTrain, extrasTrain, simple, target)
        X_valBefore, X_valAfter, Y_valBefore, Y_valAfter = self.GetFeatures(intrasVal, extrasVal, simple, target)
        X_testBefore, X_testAfter, Y_testBefore, Y_testAfter = self.GetFeatures(intrasTest, extrasTest, simple, target)


        # SINDy piecewise modeling
        differentiation_method = ps.FiniteDifference(order=2)
        feature_library = ps.PolynomialLibrary(degree=2)
        #feature_library = ps.FourierLibrary(n_frequencies=5)

        library_functions = [lambda x: np.exp(x)]
        library_function_names = [lambda x: "exp(" + x + ")"]
        custom_library = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)

        generalized_library = ps.GeneralizedLibrary([feature_library, custom_library])

        if(simple):
            generalized_library = feature_library

        # optimizer = ps.STLSQ(threshold=0.2, alpha=0.5)
        optimizer = ps.SR3(threshold=0.1, thresholder="l2")
        # optimizer = ps.SR3(trimming_fraction=0.1)
        # optimizer = ps.SSR(alpha=0.05)
        # optimizer = ps.SSR(alpha=0.05, criteria="model_residual")
        # optimizer = ps.SSR(alpha=0.05, criteria="model_residual", kappa=1e-3)
        # optimizer = ps.FROLS(alpha=0.05)
        # optimizer = ps.FROLS(alpha=0.05, kappa=1e-7)
        # optimizer = Lasso(alpha=2, max_iter=2000, fit_intercept=False)

        discrete = False

        model1 = copy.deepcopy(ps.SINDy(differentiation_method=differentiation_method,
            feature_library=generalized_library,
            optimizer=optimizer,
            discrete_time=discrete))

        model2 = copy.deepcopy(ps.SINDy(differentiation_method=differentiation_method,
            feature_library=generalized_library,
            optimizer=optimizer,
            discrete_time=discrete))

        print("Training the model ...")
        model1.fit(XBefore, t=1, x_dot=YBefore, multiple_trajectories=True)
        model2.fit(XAfter, t=1, x_dot=YAfter, multiple_trajectories=True)

        model1.print()
        model2.print()


        # Evaluate model on test data
        outBefore = np.array(model1.predict(X_testBefore, multiple_trajectories=True))
        outAfter = np.array(model2.predict(X_testAfter, multiple_trajectories=True))
        out = []
        pred = []

        overlap = len(outBefore[0]) + len(outAfter[0]) - len(extrasTrain[0])

        for i in range(len(outBefore)):
            out.append(np.concatenate((outBefore[i][:-overlap], outAfter[i]), axis=0))
            
        out = np.array(out)    
        out = np.reshape(out, out.shape[:2])

        if(target == 2):
            pred = np.cumsum(np.cumsum(out, axis=1), axis=1)
        elif(target == 1):
            pred = np.cumsum(out, axis=1)
        else:
            pred = out 


        print("MSE on test with reg_selected", mse(pred, intrasTest))
        print("MAE on test with reg_selected", mae(pred, intrasTest))
        print("DTW on test with reg_selected", dtw(pred, intrasTest))


        # Prediction plots for a random recording from the test set
        ind = random.randint(0, len(pred)-1) # 280, 391, 881, 507, 34, 1157 in train
        print(ind)
        predOut = pred[ind]

        windowSize = 20
        predOut = moving_filter(predOut, windowSize)
        predOut = moving_filter(predOut, windowSize)

        transparency = 0.7
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        fig0.tight_layout(pad=4)

        # Plot iAP prediction
        ax0.plot(predOut, linestyle='dashed')
        ax0.plot(intrasTest[ind], alpha=transparency, color='red')
        ax0.legend(['Prediction', 'Ground Truth'], prop={'size': 18})
        ax0.title.set_text("Intra-cellular AP")
        ax0.title.set_fontsize(20)
        ax0.set_xlabel("Timesteps", fontsize=20)
        ax0.set_ylabel("iAP", fontsize=20)
        # ax0.set_ylim([-0.1, 1.2])
        # ax0.tick_params(axis='x', labelsize=16)
        # ax0.tick_params(axis='y', labelsize=16)