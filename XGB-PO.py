
import numpy as np
import xgboost as xgb
import sklearn.metrics as met
from ypstruct import structure


# Cost function
def cost_function(pop, X_tr, X_te, Y_tr, Y_te):

    net = xgb.XGBRegressor(
        n_estimators=int(pop[0]),
        max_depth=int(pop[1]),
        learning_rate=pop[2] / 1000.0,
        reg_alpha=pop[3] / 1000.0,
        reg_lambda=pop[4] / 1000.0,
        objective='reg:squarederror',
        random_state=1
    )

    net.fit(X_tr, Y_tr)
    preds_tr = net.predict(X_tr)
    preds_te = net.predict(X_te)
    y = np.hstack([Y_tr, Y_te])
    preds = np.hstack([preds_tr, preds_te])
    RMSE = np.sqrt(met.mean_squared_error(y, preds))
    POP = structure(cost=RMSE, pre=preds, net=net)
    return POP.cost, POP.pre, POP.net


# Parrot Optimizer (PO)
def Parrot_Optimizer(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    lb = np.array(lb)
    ub = np.array(ub)
    # population
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))
    fitness = np.zeros(SearchAgents_no)
    Pre = [None] * SearchAgents_no
    Net = [None] * SearchAgents_no

    for i in range(SearchAgents_no):
        fitness[i], Pre[i], Net[i] = objf(X[i])
    idx = np.argmin(fitness)
    Best_pos = X[idx].copy()
    Best_score = fitness[idx]
    Best_pre = Pre[idx]
    Best_net = Net[idx]
    curve = np.zeros(Max_iter)
    
    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A = 2 * r1 - 1
            C = 2 * r2
            if np.random.rand() < 0.5:
                # exploration
                Xnew = X[i] + np.random.rand() * (Best_pos - C * X[i])
            else:
                # exploitation
                Xnew = Best_pos - A * np.abs(C * Best_pos - X[i])
            Xnew = np.clip(Xnew, lb, ub)
            fnew, prenew, netnew = objf(Xnew)
            if fnew < fitness[i]:
                X[i] = Xnew
                fitness[i] = fnew
                Pre[i] = prenew
                Net[i] = netnew
                if fnew < Best_score:
                    Best_score = fnew
                    Best_pos = Xnew.copy()
                    Best_pre = prenew
                    Best_net = netnew
        curve[t] = Best_score
        print("Iter =", t, " Best RMSE =", Best_score)
    return Best_score, Best_pos, Best_pre, Best_net, curve

# Data (same structure as your code)
X = np.loadtxt('input.txt')
Y = np.loadtxt('target.txt')
size_Y = len(Y)
size_tr = round(size_Y * 0.75)
size_te = size_Y - size_tr
X_tr = X[:size_tr]
X_te = X[size_tr:]
Y_tr = Y[:size_tr].reshape(-1)
Y_te = Y[size_tr:].reshape(-1)

# Wrapper for optimizer
def objective_wrapper(x):
    return cost_function(x, X_tr, X_te, Y_tr, Y_te)


# Problem definition
dim = 5
lb = [100,2,0.005,0,0.1]
ub = [1000,100,0.3,10,100]
SearchAgents_no = 30
Max_iter = 200


# Run Parrot Optimizer
Best_score, Best_pos, Best_pre, Best_net, curve = Parrot_Optimizer(
    objective_wrapper,
    lb,
    ub,
    dim,
    SearchAgents_no,
    Max_iter
)


print('--------------------------------------------------')
print("Best RMSE =", Best_score)
print("Best parameters =", Best_pos)

