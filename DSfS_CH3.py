# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:10:06 2021

@author: Alejandro Herrera
Data Science from scratch: Frist Principles eith Python- 
By Joel Grus
Chapters 3 & 4
"""
## CHAPTER 3: Visualizing Data
from matplotlib import pyplot as plt
import pandas as pd
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.5, 10289.7, 14958.3]

#create a line chart, years in x-axis, gdp in y-axis
plt.plot(years, gdp, color ='magenta', marker ='o', linestyle= 'solid')

# Add title:
plt.title("Nominal GDP")

# Add label to the y-axis
plt.ylabel('Billion od $')
plt.show()

#SCATTERPLOT:

test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter (test_1_grades, test_2_grades)
plt.title('Axes Aren\'t Comparable')
plt.xlabel("Test 1 Grades")
plt.ylabel("Test 2 Grade")
plt.show()

#BARCHARTS:
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

#plot bars with left x-coordinates [0, 1, 2, 3, 4] Heights[num_oscar]
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")                          # add a title 
plt.ylabel("# of Academy Awards")                  # label they y-axis

#label x-axis with movie names at bar centers
plt.xticks(range(len(movies)), movies)
plt.show()

#Visually exploring how values are distributed:

from collections import Counter
grades = [83, 95, 91, 87,70, 0, 85, 82, 100, 67, 73, 77, 0]

#Bucket grades by dectile, but put 100 in with the 90s
histogram = Counter(min(grades // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.key()],       # Shift bars right by 5
        histogram.values(),                     # Give each bar its correct height
        10,                                     # Give each bar a width of 10
        edgecolor = (0, 0, 0))                  # Black edges for each bar

plt.axis([-5, 105, 0, 5])                       # x-axis from -5 to 105,
                                                # y-axis from 0 to 5 

plt.xticks([10 * i for i in range(11)])         # x-axis labels at 0 to 100
plt.xlabel("Dectile")
plt.ylabel("# of Student")
plt.title("Distribution of Exam 1 Grades")
plt.show()


# When creating bar charts it is considered especially bad form for your
# y-axis not to start at 0
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# if you don't do this, matplotlib will label the axis 0, 1
# and then add a +2.013e3 off in the corner
plt.ticklabel(useOffdet=False)

#misleading y-axis only shows the part above 500
plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")
plt.show()

#LINE CHARTS:
variance = [1, 2, 4, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 4, 2,1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# We can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.plot(xs, variance,  'g-', label='variance')           #green solid line
plt.plot(xs, bias_squared,  'r-.', label='bias^2')        #red dot-dashed line
plt.plot(xs, variance,  'b:', label='total error')        #blue dotted line

# Because we've assigned labels to each series,
# we can get a legend for free (loc=9 means "top center")

plt.legend(loc=9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Variance Tradeoff")
plt.show()

#SCATTERPLOTS:
#Scatterplot-right choice for visualizing the relationship between two paired sets of data
#The relationship between number of friends your user have and number of
# minutes the spend on the site every day

friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [ 175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

#label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count, minute_count),               # Put the label with its point
                 xytext=(5, -5),                                # but slightly offset
                 textcoords='offset points')

plt.title("Daily Minutes vs. Number of friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("Test 1 Grade")
plt.ylabel("Test 2 Grade")
plt.show()