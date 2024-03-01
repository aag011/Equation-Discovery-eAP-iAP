import tensorflow as tf
import keras
import itertools
import numpy as np
from matplotlib import pyplot as plt, transforms
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from helper import moving_filter, mse, mae, dtw, ApplyMovingFilter


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
        gpus = tf.config.list_physical_devices('GPU')
        gpu_id = 0
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')

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


        # Estimate parameters of Physics loss equation using training data.
        # The equation is then used as a constraint in the loss function of the model.
        windowSize = 20

        v = intrasTrain
        dv = ApplyMovingFilter(np.gradient(v, axis=1), windowSize)
        d2v = ApplyMovingFilter(np.gradient(dv, axis=1), windowSize)
        d3v = ApplyMovingFilter(np.gradient(d2v, axis=1), windowSize)

        v = np.reshape(v, (-1))
        dv = np.reshape(dv, (-1))
        d2v = np.reshape(d2v, (-1))
        d3v = np.reshape(d3v, (-1))

        vFeatures = np.array([d2v*(v**2), v**3, dv*(v**2), v**4, v*d2v, v*dv, (d2v)**2, (d2v*dv), (v**3)*d2v, (v**3)*dv, v**5, (dv)**2, 
                            d2v, d3v*v, dv, np.ones(v.shape)]).T

        prediction = np.array([d3v*(v**2)]).T
        params = np.linalg.inv(vFeatures.T.dot(vFeatures)).dot(vFeatures.T.dot(prediction))


        # Linear regression model with custom loss
        class LinearReg(keras.layers.Layer):
            def __init__(self, numParams, lamda):
                super(LinearReg, self).__init__()
                self.w = self.add_weight(shape=(numParams, 1), initializer="random_normal", trainable=True)
                self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
                self.lamda = lamda
                
            def call(self, x):
                return tf.matmul(x, self.w) + self.b
            
            def loss_fn(self, output, target):
                reg = self.lamda*(tf.math.reduce_sum(tf.math.abs(self.w)) + tf.math.reduce_sum(tf.math.abs(self.b)))
                sqError = tf.math.reduce_sum(tf.math.square(output - target))
                
                d2v = tf.reshape(output, (-1)).numpy()
                dv = np.cumsum(d2v)
                v = np.cumsum(dv)
        #         d2v = np.gradient(dv)
                d3v = np.gradient(d2v)
                
                vFeatures = np.array([d2v*(v**2), v**3, dv*(v**2), v**4, v*d2v, v*dv, (d2v)**2, (d2v*dv), (v**3)*d2v, (v**3)*dv, v**5, (dv)**2, 
                            d2v, d3v*v, dv, np.ones(v.shape)]).T
                prediction = np.array([d3v*(v**2)]).T
                
                phyErr = tf.reduce_sum(tf.math.square(vFeatures.dot(params) - prediction))
        #         d1mse = tf.reduce_mean(tf.math.square(tf.math.cumsum(output, 0) - tf.math.cumsum(target, 0)))
        #         mse = tf.reduce_mean(tf.math.square(tf.math.cumsum(tf.math.cumsum(output, 0), 0) - tf.math.cumsum(tf.math.cumsum(target, 0), 0)))

                return reg + sqError + tf.cast(phyErr, tf.float32)# + d1mse + mse


        print("Training the model with all features ...")
        # Training without masks
        keras.utils.set_random_seed(812)

        X = X0
        Y = Y0
        X_val = X_val0
        Y_val = Y_val0
        X_test = X_test0
        Y_test = Y_test0

        init_lr = 1e-6
        epochs = 50
        lamdaReg = 1
        
        if(simple):
            if(target == 2):
                init_lr = 1e-4
                epochs = 50
                lamdaReg = 1
            elif(target == 1):
                init_lr = 1e-4
                epochs = 50
                lamdaReg = 1
            else:
                init_lr = 1e-4
                epochs = 50
                lamdaReg = 0.1
        else:
            if(target == 2):
                init_lr = 5e-6
                epochs = 100
                lamdaReg = 1
            elif(target == 1):
                init_lr = 1e-3
                epochs = 50
                lamdaReg = 1
            else:
                init_lr = 1e-4
                epochs = 50
                lamdaReg = 1
                
                
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=init_lr, decay_steps=10000, decay_rate=0.95)

        batch_size = len(extrasTrain[0])

        reg_selected = LinearReg(X.shape[1], lamdaReg)

        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(epochs):
            totalLoss = 0.0
            
            for i in range(0, len(X), batch_size):
                loss= 0.0
                
                with tf.GradientTape() as tape:
                    output = reg_selected(tf.convert_to_tensor(X[i:i+batch_size], dtype='float32'))
                    output = tf.reshape(output, (-1))
                    loss = reg_selected.loss_fn(output, tf.convert_to_tensor(Y[i:i+batch_size], dtype='float32'))
                    totalLoss += loss
            
                grads = tape.gradient(loss, reg_selected.trainable_weights)
                optimizer.apply_gradients(zip(grads, reg_selected.trainable_weights))
                    
            if(epoch%5==0):
                print("Epoch ", epoch, " complete with loss ", totalLoss/len(X)*batch_size)


        if(not simple):
            # Select features with high coefficient magnitude
            featureWeights = np.absolute(np.reshape(reg_selected.w.numpy(), (-1)))

            mask = featureWeights >= np.sort(featureWeights)[-10]
            print(len(featureWeights))

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

            #### Only for weight params
            # featureCombs.extend(["max"+str(i) for i in range(-50, 50)])
            # featureCombs.extend(["min"+str(i) for i in range(100)])
            ####

            print(np.array(featureCombs)[mask])

            print("Training the model with selected features ...")
            # Training with masks
            X = X0[:, mask]
            Y = Y0
            X_val = X_val0[:, mask]
            Y_val = Y_val0
            X_test = X_test0[:, mask]
            Y_test = Y_test0


            if(target == 2):
                init_lr = 5e-6
                epochs = 100
                lamdaReg = 1
            elif(target == 1):
                init_lr = 1e-6
                epochs = 100
                lamdaReg = 5
            else:
                init_lr = 1e-4
                epochs = 50
                lamdaReg = 1

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=init_lr, decay_steps=10000, decay_rate=0.95)

            batch_size = len(extrasTrain[0])

            reg_selected = LinearReg(X.shape[1], lamdaReg)

            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            for epoch in range(epochs):
                totalLoss = 0.0

                for i in range(0, len(X), batch_size):
                    loss= 0.0

                    with tf.GradientTape() as tape:
                        output = reg_selected(tf.convert_to_tensor(X[i:i+batch_size], dtype='float32'))
                        output = tf.reshape(output, (-1))
                        loss = reg_selected.loss_fn(output, tf.convert_to_tensor(Y[i:i+batch_size], dtype='float32'))
                        totalLoss += loss

                    grads = tape.gradient(loss, reg_selected.trainable_weights)
                    optimizer.apply_gradients(zip(grads, reg_selected.trainable_weights))

                if(epoch%5==0):
                    print("Epoch ", epoch, " complete with loss ", totalLoss/len(X)*batch_size)


        # Evaluate on validation data
        valOut = reg_selected(tf.convert_to_tensor(X_val, dtype='float32'))

        if(target == 2):
            predd2_selected = tf.reshape(valOut, (-1, 8000)).numpy()
            predd1_selected = np.cumsum(predd2_selected, axis=1)
            prediAP_selected = np.cumsum(predd1_selected, axis=1)
        elif(target == 1):
            predd1_selected = tf.reshape(valOut, (-1, 8000)).numpy()
            prediAP_selected = np.cumsum(predd1_selected, axis=1)
        else:
            prediAP_selected = tf.reshape(valOut, (-1, 8000)).numpy()
        

        print("MSE on val with reg_selected", mse(prediAP_selected, intrasVal))
        print("MAE on val with reg_selected", mae(prediAP_selected, intrasVal))
        print("DTW on val with reg_selected", dtw(prediAP_selected, intrasVal))


        # Evaluate model on test data
        testOut = reg_selected(tf.convert_to_tensor(X_test, dtype='float32'))

        if(target == 2):
            predd2_selected = tf.reshape(testOut, (-1, 8000)).numpy()
            predd1_selected = np.cumsum(predd2_selected, axis=1)
            prediAP_selected = np.cumsum(predd1_selected, axis=1)
        elif(target == 1):
            predd1_selected = tf.reshape(testOut, (-1, 8000)).numpy()
            prediAP_selected = np.cumsum(predd1_selected, axis=1)
        else:
            prediAP_selected = tf.reshape(testOut, (-1, 8000)).numpy()

        print("MSE on test with reg_selected", mse(prediAP_selected, intrasTest))
        print("MAE on test with reg_selected", mae(prediAP_selected, intrasTest))
        print("DTW on test with reg_selected", dtw(prediAP_selected, intrasTest))



        # Prediction plots
        ind = random.randint(0, len(extrasTest)-1)#54, 12, 40, 308, 202
        print(ind)


        transparency = 0.7
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        fig0.tight_layout(pad=5)

        # Plot iAP
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