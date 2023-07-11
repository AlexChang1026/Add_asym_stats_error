import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

## The addition of asymmetric stats uncertainty is based on Variable Gaussian as a approximated log likelihood function
## The whole calculation is from Barlow's publication: https://arxiv.org/abs/physics/0406120v1
## This version is only for asymmetric statistical uncertainty addtion

##################### Implement procedure ################################

### val1^{+errorh1}_{-errorl1} + val2^{+errorh2}_{-errorl2} ....
### All you need to do is listing

### x0 = [val1,val2,...] values array
### error_up = [errorh1,errorh2...] error+ array
### error_low = [errorl1,errorl2...] error- array
### add_asym_err(x0,error_up,error_low)


### then return:

### [value, combined_error+,combined_error- ]
############################################################################



def sum_log_llh (u,x0,errh,errl):  ### buid an approximated log liklihood function to approach a ture distribution of x1+x2

   sig1 = 2*errh[0]*errl[0]/(errh[0]+errl[0])
   sig1_prime = (errh[0]-errl[0])/(errh[0]+errl[0])
   
   sig2 = 2*errh[1]*errl[1]/(errh[1]+errl[1])
   sig2_prime = (errh[1]-errl[1])/(errh[1]+errl[1])
   
   x1 = 0.
   
   while True:    ### iteration to find a corresponding x1
   
         x1_old = x1
    
         w1 = (sig1+sig1_prime*(x1))**3/(2*sig1)
         w2 = (sig2+sig2_prime*(u-x1))**3/(2*sig2)
   
         x1 = u*(w1/(w1+w2))
         if x1_old != 0.:
            if abs(x1/x1_old-1.) <1e-5 : break
  
   log_llh = -0.5*(((x1)/(sig1+sig1_prime*(x1)))**2 + ((u-x1)/(sig2+sig2_prime*(u-x1)))**2 ) +0.5
   
   return log_llh
   
def asym_err_finder(x0,errh,errl):  ####### add first two value with asymmetric uncertainties
 
   mean = np.sum(x0);
   error_hi = float(scipy.optimize.fsolve(sum_log_llh,1*errl[0],(x0,errh,errl))) ##### find error+ and error- by \delta ln(L) = 1/2
   error_lo = float(scipy.optimize.fsolve(sum_log_llh,-1*errl[0],(x0,errh,errl)))
   
   return np.array([mean,error_hi,error_lo])


def add_asym_err(x0,errh,errl):  ####### add several values with asymmetric errors by using asym_err_finder()
 
    for i in range(len(x0)-1):
    
        x0_first_2 = x0[:2]
        errh_first_2 = errh[:2]
        errl_first_2 = errl[:2]
        
        sol = asym_err_finder(x0_first_2,errh_first_2,errl_first_2)
   
        x0 = np.delete(x0,[0,1])
        errh = np.delete(errh,[0,1])
        errl = np.delete(errl,[0,1])
        
        x0 = np.append(x0,sol[0])
        errh = np.append(errh,sol[1])
        errl = np.append(errl,-1*sol[2])
        
    return np.array([x0,errh,-1*errl])


########################################################################################


