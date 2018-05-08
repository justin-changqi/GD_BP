import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

class BackPropagation:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df['class'] = df['class'].apply(lambda x: 0 if x == 'Iris-setosa' \
                                                    else (1 if x == 'Iris-versicolor' else 2))
        self.irisdata = np.array(df.astype(float).values.tolist()[50:])

    def randomSplitData(self, train_size=0.7):
        random_data = self.irisdata.copy()
        np.random.shuffle(random_data)
        num_train = math.ceil(train_size*len(random_data))
        num_test = len(random_data) - num_train
        self.train = random_data[:num_train, [2, 3, 4]]
        self.test = random_data[num_train:, [2, 3, 4]]
        self.mean = np.mean(self.train, axis=0)
        self.sd = np.std(self.train, axis=0)

    def training(self, train_data, epoche=1000, learning_rate=0.1):
        self.w =  np.random.rand(8)
        self.eta = learning_rate
        errors = []
        for i in range(epoche):
            error = 0
            for data in train_data:
                y1, y2 = self.forward(data)
                if (data[-1] == 1):
                    d1 = 0.9
                    d2 = 0.1
                else:
                    d1 = 0.1
                    d2 = 0.9
                error += pow(y1-d1, 2)/2 + pow(y2-d2, 2)/2
                self.updateWeight(y1, d1, y2, d2)
            errors.append(error/len(train_data))
        return errors

    def forward(self, x):
        x1, x2 = x[0], x[1]
        w = self.w
        q1 = x1*w[0] + x2*w[1]
        q2 = x1*w[2] + x2*w[3]
        h1 = self.sigmoid(q1)
        h2 = self.sigmoid(q2)
        z1 = h1*w[4] + h2*w[5]
        z2 = h1*w[6] + h2*w[7]
        y1 = self.sigmoid(z1)
        y2 = self.sigmoid(z2)
        self.x1, self.x2, self.q1, self.q2 = x1, x2, q1, q2
        self.h1, self.h2, self.z1, self.z2 = h1, h2, z1, z2
        return y1, y2

    def updateWeight(self, y1, d1, y2, d2):
        w = self.w
        eta = self.eta
        e1_dz1 = (y1-d1)*y1*(1-y1)
        e2_dz2 = (y2-d2)*y2*(1-y2)
        z1_dq1 = w[4]*self.q1*(1-self.q1)
        z1_dq2 = w[5]*self.q2*(1-self.q2)
        z2_dq1 = w[6]*self.q1*(1-self.q1)
        z2_dq2 = w[7]*self.q2*(1-self.q2)
        dw1 = e1_dz1*z1_dq1*self.x1 + e2_dz2*z2_dq1*self.x1
        dw2 = e1_dz1*z1_dq1*self.x2 + e2_dz2*z2_dq1*self.x2
        dw3 = e1_dz1*z1_dq2*self.x1 + e2_dz2*z2_dq2*self.x1
        dw4 = e1_dz1*z1_dq2*self.x2 + e2_dz2*z2_dq2*self.x2
        dw5 = e1_dz1*self.h1
        dw6 = e1_dz1*self.h2
        dw7 = e2_dz2*self.h1
        dw8 = e2_dz2*self.h2
        self.w = np.array([w[0]-eta*dw1, w[1]-eta*dw2, w[2]-eta*dw3, w[3]-eta*dw4, \
                           w[4]-eta*dw5, w[5]-eta*dw6, w[6]-eta*dw7, w[7]-eta*dw8])

    def sigmoid(self, x):
        return 1. / (1. + math.exp(-x))

    def testing(self, test):
        corect = 0
        for data in test:
            y1, y2 = self.forward(data)
            result = 0
            if y1 >= y2:
                result = 1
            else:
                result = 2
            if result == data[-1]:
                corect += 1
        return corect/len(test)

    def zScoreNormalize(self, dataset):
        for data in dataset:
            for i in range(len(data)-1):
                data[i] = (data[i] - self.mean[i]) / self.sd[i]

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    bp = BackPropagation('iris_data_set/iris.data')
    bp.randomSplitData(train_size=0.7)
    bp.zScoreNormalize(bp.train)
    bp.zScoreNormalize(bp.test)
    errors = bp.training(bp.train, epoche=10000, learning_rate=0.01)
    print ('Accuracy:', bp.testing(bp.test))
    plt.plot(errors)
    plt.ylabel('MSE')
    plt.xlabel('epoche')
    plt.show()
