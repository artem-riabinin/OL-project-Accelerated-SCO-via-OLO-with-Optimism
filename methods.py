import numpy as np
from numpy import linalg as la
import random



def AccelPDA(cost, grad, x0, T, d, batch, alpha, D, L, eta, r):      
  
    x = [x0 for i in range(T+1)]
    x_aver = [x0 for i in range(T+1)]
    G = np.zeros((T+1, d))
    G[0,:] = grad(x0, batch[0])
    sum_alpha_x = x0
    sum_alpha = alpha[1]
    sum_alpha_g = 0
    f = np.zeros(T+1)
    f[0] = cost(x0)
    
    for t in range(1,T+1): 
        eta_t = 4*L + eta * alpha[t+1] * np.sqrt(t+1)
        sum_alpha_g += alpha[t] * G[t-1]
        z_t = (-1) * (sum_alpha_g + alpha[t+1] * G[t-1]) / eta_t
        if np.linalg.norm(z_t) >= r:
            x[t] = z_t / np.linalg.norm(z_t) * r
        else: x[t] = z_t
        sum_alpha += alpha[t+1]
        sum_alpha_x = sum_alpha_x + alpha[t+1]*x[t]
        x_aver[t] = sum_alpha_x / sum_alpha
        G[t] = grad(x_aver[t], batch[t])
        f[t] = cost(x_aver[t])
     
    return x_aver, f



def UniXGrad(cost, grad, y0, T, d, batch, alpha, r, D):      
  
    x = [y0 for i in range(T+1)]
    x_aver = [y0 for i in range(T+1)]
    y = [y0 for i in range(T+1)]
    G = np.zeros((T+1, d))
    M = np.zeros((T+1, d))
    sum_alpha_x = 0
    sum_alpha = 0
    sum_g = 0
    f = np.zeros(T+1)
    f[0] = cost(y0)
    
    for t in range(1,T+1):  
        sum_alpha_x += alpha[t-1]*x[t-1]
        sum_alpha += alpha[t]  
        z_t = (alpha[t]*y[t-1] + sum_alpha_x) / sum_alpha
        M[t] =  grad(z_t, batch[t-1])
        sum_g += alpha[t-1]**2 * np.linalg.norm(G[t-1] - M[t-1])**2 
        eta_t = 2*D / np.sqrt(1 + sum_g)
        if np.linalg.norm(y[t-1] - eta_t*alpha[t]*M[t]) >= r:
            x[t] = (y[t-1] - eta_t*alpha[t]*M[t]) / np.linalg.norm((y[t-1] - eta_t*alpha[t]*M[t])) * r
        else: x[t] = (y[t-1] - eta_t*alpha[t]*M[t])
        x_aver[t] = (alpha[t]*x[t] + sum_alpha_x) / sum_alpha
        G[t] = grad(x_aver[t], batch[t-1])
        if np.linalg.norm(y[t-1] - eta_t*alpha[t]*G[t]) >= r:
            y[t] = (y[t-1] - eta_t*alpha[t]*G[t]) / np.linalg.norm((y[t-1] - eta_t*alpha[t]*G[t])) * r
        else: y[t] = (y[t-1] - eta_t*alpha[t]*G[t])
        f[t] = cost(x_aver[t])
        
    return x_aver, f



def UniAdaGrad(cost, grad, x0, T, d, batch, alpha, r, R):      
  
    gamma = 2/R
    x = [x0 for i in range(T+1)]
    x_aver = [x0 for i in range(T+1)]
    G = np.zeros((T+1, d))
    G[0,:] = grad(x0, batch[0])
    Z = np.zeros((T+1, d))
    sum_alpha_x = x0
    sum_alpha = alpha[1]
    eta = np.zeros((T+1, d))
    sum_alpha_g_2 = np.zeros(d)
    for j in range(d):
        sum_alpha_g_2[j] += alpha[1]**2 * (G[0,j])**2
        eta[0,j] = np.sqrt(sum_alpha_g_2[j])
    sum_eta_x = np.zeros(d)
    sum_alpha_g = np.zeros(d)
    sum_eta_eta = np.zeros(d)
    for j in range(d):
        sum_eta_x[j] += eta[0,j] * x[0][j]
        sum_alpha_g[j] += alpha[1] * G[0,j]
        sum_eta_eta[j] += eta[0,j]
    f = np.zeros(T+1)
    f[0] = cost(x0)
    
    for t in range(1,T+1):  
        for j in range(d):
            if t>1:
                sum_eta_x[j] += (eta[t-1,j] - eta[t-2,j]) * x[t-1][j]
                sum_alpha_g[j] += alpha[t] * G[t-1,j]
                sum_eta_eta[j] += eta[t-1,j] - eta[t-2,j]
                Z[t,j] = (gamma*sum_eta_x[j] - sum_alpha_g[j] - alpha[t+1]*G[t-1,j]) / (gamma*sum_eta_eta[j])
        if np.linalg.norm(Z[t,j]) >= r:
            x[t] = Z[t,:] / np.linalg.norm(Z[t,:]) * r
        else: x[t] = Z[t,:]
        sum_alpha += alpha[t+1]
        sum_alpha_x = sum_alpha_x + alpha[t+1]*x[t]
        x_aver[t] = sum_alpha_x / sum_alpha
        G[t,:] = grad(x_aver[t], batch[t])
        for j in range(d):
            sum_alpha_g_2[j] += alpha[t+1]**2 * (G[t,j] - G[t-1,j])**2
            eta[t,j] = np.sqrt(sum_alpha_g_2[j])
        f[t] = cost(x_aver[t])
          
    return x_aver, f

def AdaGrad(cost, grad, x0, T, d, batch, D, r):      
  
    x = [x0 for i in range(T+1)]
    x_aver = [x0 for i in range(T+1)]
    sum_x = grad(x_aver[0], batch[0])
    G = np.zeros((T+1, d))
    G[0,:] = grad(x0, batch[0])
    sum_g_2 = np.linalg.norm(G[0,:])**2
    f = np.zeros(T+1)
    f[0] = cost(x0)
    
    for t in range(1,T+1): 
        
        eta_t = D / np.sqrt(2 * sum_g_2)
        
        z_t = x[t-1] - eta_t * G[t-1,:]
        
        if np.linalg.norm(z_t) >= r:
            x[t] = z_t / np.linalg.norm(z_t) * r
        else: x[t] = z_t
            
        sum_x += x[t]
        x_aver[t] = sum_x / (t + 1)
        G[t,:] = grad(x_aver[t], batch[t])
        sum_g_2 += np.linalg.norm(G[t,:])**2
        f[t] = cost(x_aver[t])
     
    return x_aver, f