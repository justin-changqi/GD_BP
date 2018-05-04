import numpy as np
import random
import math

def getx(theta):
    return math.cos(theta)

def gety(theta):
    return math.sin(theta)

def f(theta):
    return math.cos(theta)+math.sin(theta)

def df(theta):
    return math.cos(theta)-math.sin(theta)

def GradientDecent(eta, max_iterative=1000):
    theta = random.uniform(-10., 10.)
    for i in range(max_iterative):
        theta = theta + eta*df(theta)
    return (getx(theta), gety(theta))

if __name__ == '__main__':
    print ('Learning Rate 1,    max iterative 10:  ', GradientDecent(5,max_iterative=10))
    print ('Learning Rate 0.1,  max iterative 10:  ', GradientDecent(0.1,max_iterative=10))
    print ('Learning Rate 0.01, max iterative 10:  ', GradientDecent(0.01,max_iterative=10))
    print ('Learning Rate 1,    max iterative 100: ', GradientDecent(5,max_iterative=100))
    print ('Learning Rate 0.1,  max iterative 100: ', GradientDecent(0.1,max_iterative=100))
    print ('Learning Rate 0.01, max iterative 100: ', GradientDecent(0.01,max_iterative=100))
    print ('Learning Rate 1,    max iterative 1000:', GradientDecent(5,max_iterative=1000))
    print ('Learning Rate 0.1,  max iterative 1000:', GradientDecent(0.1,max_iterative=1000))
    print ('Learning Rate 0.01, max iterative 1000:', GradientDecent(0.01,max_iterative=1000))
