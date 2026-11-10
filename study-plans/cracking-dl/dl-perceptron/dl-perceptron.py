import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.int64); weights=np.zeros(X.shape[1],dtype=np.float64); bias=0.0
    for _ in range(epochs):
        for point,target in zip(X,y):
            prediction=1 if point@weights+bias>=0.0 else 0; error=target-prediction; weights+=lr*error*point; bias+=lr*error
    return [round(float(v),4) for v in weights],round(float(bias),4)
