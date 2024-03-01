# Equation Discovery to relate Intra-cellular and Extra-cellular Membrane Potentials

## Setting up the environment

Run the following command to set up the environment for this experiment

```conda env create -f environment.yml```

## Run the experiment

To run the experiment, use the following command

``` python run.py --model=<model name> --target=<target type> --feature=<feature type> ```

Model name can be $\rightarrow$ [Lasso, SINDy, LassoPiecewise, SINDyPiecewise, Physics]

Target type is 0, 1, or 2 where
* If target is 0, the model will predict iAP
* If target is 1, the model will predict the first derivative of iAP and iAP will be computed from the first derivative using integration (cumulative sum)
* If target is 2, the model will predict the second derivative of iAP and iAP will be computed from the second derivative using integration (cumulative sum)

Feature type can be $\rightarrow$ ['simple', 'complex']

Simple feature set, $$F_{simple} = \left \\{ \frac{d^2\ eAP}{dt^2},\ \frac{d\ eAP}{dt},\ eAP, t \right \\}$$
Complex feature set, $$F_{complex} = \left \\{ \frac{d^2\ eAP}{dt^2},\ \frac{d\ eAP}{dt},\ eAP,\ e^{\frac{d^2\ eAP}{dt^2}},\ e^{\frac{d\ eAP}{dt}},\ e^{eAP}, t \right \\}$$

Following is an example command that runs LassoPiecewise model to predict the first derivative of iAP using the complex feature set:

``` python run.py --model='LassoPiecewise' --target=1 --feature='complex' ```
