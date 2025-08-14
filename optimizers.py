import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model, grads):
        dW1, db1, dW2, db2 = grads
        model.W1 -= self.lr * dW1
        model.b1 -= self.lr * db1
        model.W2 -= self.lr * dW2
        model.b2 -= self.lr * db2

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, model, grads):
        dW1, db1, dW2, db2 = grads
        if not self.m:
            self.m = {k: np.zeros_like(v) for k,v in vars(model).items() if k.startswith("W") or k.startswith("b")}
            self.v = {k: np.zeros_like(v) for k,v in vars(model).items() if k.startswith("W") or k.startswith("b")}

        self.t += 1
        for param, grad in zip(["W1","b1","W2","b2"], grads):
            self.m[param] = self.beta1*self.m[param] + (1-self.beta1)*grad
            self.v[param] = self.beta2*self.v[param] + (1-self.beta2)*(grad**2)

            m_hat = self.m[param] / (1 - self.beta1**self.t)
            v_hat = self.v[param] / (1 - self.beta2**self.t)

            vars(model)[param] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
