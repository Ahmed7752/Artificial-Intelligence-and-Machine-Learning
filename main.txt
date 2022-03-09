# Author: Ahmed Mohamed, ID: 18124225

# a machine learning model to predict if a mushroom is poisonous or edible.

# to manipulate data as a dataframe
import pandas as pd
# import numpy as np
import numpy as np
# to preprocess data
from sklearn import preprocessing
# to visualise data and results
import matplotlib.pyplot as plt
# to split datain to training and test datasets
from sklearn.model_selection import train_test_split
# to change string values into ints
from sklearn.preprocessing import LabelEncoder
# to modelgaussian naive bayes classifier
from sklearn.naive_bayes import GaussianNB
# to calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
# Seaborn visualization library
import seaborn as sns
# to visualise accuracy score
from sklearn.metrics import confusion_matrix

# import dataset.
data = pd.read_csv('mushroms.csv', header=None)


# use LabelEncoder()from scikit-learnlibrary to convert string feature values to ints.
def Encoder(data):
    le = LabelEncoder()
    # Loop though all columns in the dataset.
    for feature in data:
        try:
            # Fit label encoder and return encoded labels.
            data[feature] = le.fit_transform(data[feature])
        except:
            print('Error encoding ' + feature)
    return data


data = Encoder(data)

# Histogram graph for the Distribution of edible and poisonous mushrooms.
names = ['edible', 'poisonous']
# initiate figure.
fig = plt.figure()
# set hist dimensions.
ax = fig.add_subplot(1, 1, 1)
# Set amount of bins and their properties.
ax.hist(data[0], bins= range(3), rwidth=0.8, align='left', color='red')
# set frequency of ticks on the x axis.
plt.xticks(ticks=np.arange(2), labels=names)
# set titles and labels.
plt.title('A histogram to show the distribution of '
          '\n'
          'edible and poisonous mushrooms in dataset')
plt.xlabel('class of muchrooms')
plt.ylabel('Amount of mushrooms')
# Save hist as a png.
plt.savefig("e-p-distribution.png")
# Display hist.
plt.show()

# Histogram graph for the Distribution of mushroom odors.
# Return a Series containing counts of unique values.
odor_data = data[5].value_counts()
# initiate figure.
fig = plt.figure()
# set hist dimensions.
ax = fig.add_subplot(1, 1, 1)
# Set amount of bins and their properties.
ax.hist(odor_data, bins=9, rwidth=0.8, color='red')
# set titles and labels.
plt.title('A histogram to show the distribution of '
          '\n'
          'odors of all mushrooms in dataset ')
plt.xlabel('Amount of mushrooms')
plt.ylabel('Distribution of odors')
# Save hist as a png.
plt.savefig("odors-distribution.png")
# Display hist.
plt.show()

# Histogram graph for the Distribution of mushroom gill-colors.
# Return a Series containing counts of unique values.
gill_colors_data = data[9].value_counts()
# initiate figure.
fig = plt.figure()
# set hist dimensions.
ax = fig.add_subplot(1, 1, 1)
# Set amount of bins and their properties.
ax.hist(gill_colors_data, bins=10, rwidth=0.8, color='red')
# set titles and labels.
plt.title('A histogram to show the distribution of '
          '\n '
          'gill-colors on all mushrooms in dataset ')
plt.xlabel('Amount of mushrooms')
plt.ylabel('Distribution of gill-colors')
# Save hist as a png.
plt.savefig("gill-color-distribution.png")
# Display hist.
plt.show()

# Create the  pairplots for all data in the dataset and select the required features.
data_plot1 = sns.pairplot(data, vars=[1,2,3,4,5,6,7])
data_plot2 = sns.pairplot(data, vars=[8,9,10,11,12,13,14])
data_plot3 = sns.pairplot(data, vars=[15,16,17,18,19,20,21,22])
# Save all pairplots as a png file format.
data_plot1.savefig("seaborn-pairplot-1.png") #Save the graph as a png
data_plot2.savefig("seaborn-pairplot-2.png") #Save the graph as a png
data_plot3.savefig("seaborn-pairplot-3.png") #Save the graph as a png
# Display all pariplots.
plt.show()


# data slicing
# Select all rows for all column expect column at header 0 as that will be use for the target.
features = data.iloc[:, 1:]
# Select all rows for the column at header 0, which shows if the mushroom is poisonous or edible.
target = data.iloc[:, 0]
# split the data into training and testing sets.
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            # Split training and testing size 67% to 33%
                                                                            target, test_size=0.33, random_state=10)

# building model using Gaussian Naive Bayes algorithm.
model = GaussianNB()
# Fit Gaussian Naive Bayes according to features_train and target_train.
model.fit(features_train, target_train)
# Perform classification.
target_pred = model.predict(features_test)

# print accuracy score
print(accuracy_score(target_test, target_pred, normalize=True) * 100)

# create a matrix with the prediction data from the model
cf_matrix = confusion_matrix(target_test, target_pred)
# create a heat map with the seaborn library for the matrix
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')
# save heat map as a png file format.
plt.savefig("e-p-predictions.png")
# Display the heat map.
plt.show()
