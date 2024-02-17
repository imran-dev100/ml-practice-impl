import numpy as np
import csv
import matplotlib.pyplot as plt

file='./Housing-2.csv'

x_train = np.array([]) # Area of the house
y_train = np.array([]) # Price

with open(file, mode ='r')as file:
  csvFile = csv.reader(file)
  header = next(csvFile) # skipping headers
  
  for line in csvFile:
        x_train = np.append(x_train, float(line[1]))
        y_train = np.append(y_train, float(line[0]))

m = x_train.shape[0]
n = len(y_train)

#print(f"m - length of x_train = {m}")
#print(f"n - length of y_train= {n}")

for i in range(m):
    print(f"x^{i}, y^{i} = {x_train[i]}, {y_train[i]}")

plt.scatter(x_train, y_train, marker = 'x')
plt.title("Housing prices")
plt.ylabel("Price")
plt.xlabel("Area")
plt.show()


w = 1000
b = 100
print(f"w: {w}\nb: {b}")


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

temp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, temp_f_wb, c = 'b', label = 'Our Prediction')
plt.scatter(x_train, y_train, marker = 'x', label = 'Actual Value')
plt.title("Housing prices")
plt.ylabel("Price")
plt.xlabel("Area")
plt.legend()
plt.show()


x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"cost_1200sqft: {cost_1200sqft}")