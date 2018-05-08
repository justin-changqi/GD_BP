import math

if __name__ == "__main__":
    eta0 = 0.1
    sd0 = 0.1
    _lambda = 10
    t=0.
    result = 0
    while True:
        sd = sd0 * math.exp(t/10.)
        result = eta0*math.exp(-t/_lambda)*math.exp(1./(pow(sd,2)*2))
        t += 1
        if (result < 0.001):
            break
    print ('min iterative: ', t)
    print ('learning rate: ', result)
