#import model
import numpy as np
import matplotlib.pyplot as plt

#import train and test true values
true_y = np.loadtxt('true_Test_smooth.txt')
true_y1 = np.loadtxt('true_Train_smooth.txt')

#import test data of models to plot
y1 = np.loadtxt('model1_predTest.txt')
y2 = np.loadtxt('model2_predTest.txt')
y3 = np.loadtxt('model3_predTest.txt')
y4 = np.loadtxt('model4_predTest.txt')
y5 = np.loadtxt('model5_predTest.txt')
y6 = np.loadtxt('model6_predTest.txt')
y7 = np.loadtxt('model7_predTest.txt')
y8 = np.loadtxt('model8_predTest.txt')

#import train data of models to plot
y11 = np.loadtxt('model1_predTrain.txt')
y21 = np.loadtxt('model2_predTrain.txt')
y31 = np.loadtxt('model3_predTrain.txt')
y41 = np.loadtxt('model4_predTrain.txt')
y51 = np.loadtxt('model5_predTrain.txt')
y61 = np.loadtxt('model6_predTrain.txt')
y71 = np.loadtxt('model7_predTrain.txt')
y81 = np.loadtxt('model8_predTrain.txt')

i = 5

print y1
plt.plot(true_y[-5,-9:-1], ls='--', linewidth=2, color='tomato')
plt.plot(y1[-5,-9:-1], ls='--', linewidth=2)
plt.plot(y2[-5,-9:-1], ls='--', linewidth=2)
plt.plot(y3[-5,-9:-1], ls='--', linewidth=2)
plt.plot(y4[-5,-9:-1], ls='--', linewidth=2)
plt.plot(y5[-5,-9:-1], ls='--', linewidth=2)
#plt.plot(y6[5,-9:-1], ls='--', linewidth=2)
plt.plot(y7[-5,-9:-1], ls='--', linewidth=2)
plt.plot(y8[-5,-9:-1], ls='--', linewidth=2)
plt.autoscale(enable=True, axis='y')
plt.show()


