import csv
import numpy as np
import matplotlib.pyplot as plt
# from lab_utils_multi import  load_house_data, run_gradient_descent 
# from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
# from lab_utils_common import dlc
np.set_printoptions(precision=2)
# plt.style.use('./deeplearning.mplstyle')



def load_house_data():
    file='/Volumes/ExternalSSD/ML-Specialization/jupyter-workspace/ml-practice-impl/sample-data/Housing-2.csv'

    x_array = [] # Area of the house in 1000 sq. ft.
    # print(f"Empty Array initialization:{x_array}")
    y_train = np.array([]) # Price of the house in 1000 USD

    with open(file, mode ='r')as file:
        csvFile = csv.reader(file)
        header = next(csvFile) # skipping headers
  
        for line in csvFile:
            # print(f"{int(line[1])} - {int(line[2])} - {int(line[3])} - {int(line[4])}")
            x = [int(line[1]) / float(1000) ,  # Area
                int(line[2]),  # Bedrooms
                int(line[4]),  # Floors
                int(line[3])]  # Bathrooms
            x_array.append(x)
            y_train = np.append(y_train, float(line[0])/1000)

    x_train = np.array(x_array)

    print(f"x_train: {x_train}")
    print(f"y_train: {y_train}")
    return x_train, y_train

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors', 'bathrooms']


fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True) 
#fig,ax=plt.subplots(1, 4, figsize=(24, 6), sharey=True) 

for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()