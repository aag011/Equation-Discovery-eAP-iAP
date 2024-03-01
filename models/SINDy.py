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
        X = []
        Y = []
        windowSize = 20
        
        t = np.array([i for i in range(len(extras[0]))])/len(extras[0])
        
        for i in range(len(extras)):
            iAP = intras[i]
            diAP = np.array(moving_filter(np.gradient(iAP), windowSize))
            d2iAP = np.array(moving_filter(np.gradient(np.gradient(iAP)), windowSize))

            eAP = extras[i]
            deAP = np.array(moving_filter(np.gradient(eAP), windowSize))
            d2eAP = np.array(moving_filter(np.gradient(deAP), windowSize))


            featureMatrix = [d2eAP, deAP, eAP, t]

    #         maxInd = np.argmax(eAP[:1500])
    #         minInd = 1500 + np.argmin(eAP[1500:])

    #         distFromMax = [i for i in range(len(extras[0]))]
    #         distFromMin = [i for i in range(len(extras[0]))]

    #         distFromMax = np.exp(-np.absolute(distFromMax - maxInd))
    #         distFromMin = np.exp(-np.absolute((distFromMin - minInd)/50))

    #         featureList.append(distFromMax)
    #         featureList.append(distFromMin)
            
            X.append(np.array(featureMatrix).T)

            if(target == 0):
                Y.append(iAP)
            elif(target == 1):
                Y.append(diAP)
            else:
                Y.append(d2iAP)
        
        return X, Y


    def RunModel(self, intrasTrain, intrasVal, intrasTest, extrasTrain, extrasVal, extrasTest, simple, target):
        print("Extracting features ...")
        
        # Extract features from train, val and test data
        X0, Y0 = self.GetFeatures(intrasTrain, extrasTrain, simple, target)
        X_val0, Y_val0 = self.GetFeatures(intrasVal, extrasVal, simple, target)
        X_test0, Y_test0 = self.GetFeatures(intrasTest, extrasTest, simple, target)


        # SINDy model
        differentiation_method = ps.FiniteDifference(order=3)
        feature_library = ps.PolynomialLibrary(degree=2)
        #feature_library = ps.FourierLibrary(n_frequencies=5)

        library_functions = [
            lambda x: np.exp(x),
        #     lambda x: 1.0 / (1 + x),
        #     lambda x: x,
        #     lambda x, y: np.sin(x + y),
        ]

        library_function_names = [
            lambda x: "exp(" + x + ")",
        #     lambda x: "1/(1 + " + x + ")",
        #     lambda x: x,
        #     lambda x, y: "sin(" + x + "," + y + ")",
        ]

        custom_library = ps.CustomLibrary(
            library_functions=library_functions, function_names=library_function_names
        )

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
        # optimizer = Lasso(alpha=0.000001)#, max_iter=2000)

        discrete = False

        model = copy.deepcopy(ps.SINDy(differentiation_method=differentiation_method,
            feature_library=generalized_library,
            optimizer=optimizer,
            discrete_time=discrete))

        print("Training the model ...")
        model.fit(X0, t=1, x_dot=Y0, multiple_trajectories=True)

        model.print()


        # A small digression to experiment with predicting residuals in iAP (inspired by gradient boosting). Can be ignored for reproducing numbers
        # predD1 = np.array(model.predict(X0, multiple_trajectories=True))
        # predD1 = np.reshape(predD1, (predD1.shape[:2]))

        # t = np.array([i for i in range(len(extrasTrain[0]))])/len(extrasTrain[0])
        # predD1 = [np.array([x, t]).T for x in predD1]

        # model2 = copy.deepcopy(ps.SINDy(differentiation_method=differentiation_method,
        #     feature_library=generalized_library,
        #     optimizer=optimizer,
        #     discrete_time=discrete))


        # r = Y0 - predD1
        # model2.fit(X0, t=1, x_dot=[x for x in r], multiple_trajectories=True)

        # model2.print()


        # Evaluate model on test data
        out = np.array(model.predict(X_test0, multiple_trajectories=True))
        out = np.reshape(out, (out.shape[:2]))
        prediAP = []

        if(target == 2):
            prediAP = np.cumsum(np.cumsum(out, axis=1), axis=1)
        elif(target == 1):
            prediAP = np.cumsum(out, axis=1)
        else:
            prediAP = out


        # res = np.array(model2.predict(X_test0, multiple_trajectories=True))
        # res = np.reshape(res, res.shape[:2])

        # predD1 += res
        # predD1 = [np.array([x, t]).T for x in predD1]

        print("MSE on test with reg_selected", mse(prediAP, intrasTest))
        print("MAE on test with reg_selected", mae(prediAP, intrasTest))
        print("DTW on test with reg_selected", dtw(prediAP, intrasTest))


        # Prediction plots for a random recording from the test set
        ind = random.randint(0, len(X_test0)-1)#126, 311, 268, 280, 310 ---- 266, 283, 310, 186, 4 --- 319, 274
        print(ind)
        

        transparency = 0.7
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        fig0.tight_layout(pad=5)

        # Plot iAP
        ax0.plot(prediAP[ind], linestyle='dashed')
        ax0.plot(intrasTest[ind], alpha=transparency, color='red')
        ax0.legend(['Prediction', 'Ground Truth'], prop={'size': 18})
        ax0.title.set_text("Intra-cellular AP")
        ax0.title.set_fontsize(20)
        ax0.set_xlabel("Timesteps", fontsize=20)
        ax0.set_ylabel("iAP", fontsize=20)
        # ax0.set_ylim([-0.3, 1.1])
        # ax0.tick_params(axis='x', labelsize=16)
        # ax0.tick_params(axis='y', labelsize=16)