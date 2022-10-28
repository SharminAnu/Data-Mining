
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:50:11 2021

@author: ss3727s
"""

import pandas as pd
import numpy as np
from collections import Counter
from pprint import pprint


"""
Part 1 of Assignment2: Implement the ID3 decision tree induction algorithm by using given dataset
"""

"""
Load the given dataset into a dataframe.
"""

level = 'Senior,Senior,Mid,Junior,Junior,Junior,Mid,Senior,Senior,Junior,Senior,Mid,Mid,Junior'.split(',')
lang = 'Java,Java,Python,Python,R,R,R,Python,R,Python,Python,Python,Java,Python'.split(',')
tweets = 'no,no,no,no,yes,yes,yes,no,yes,yes,yes,no,yes,no'.split(',')
phd = 'no,yes,no,no,no,yes,yes,no,no,no,yes,yes,no,yes'.split(',')
hire = 'False,False,True,True,True,False,True,False,True,True,True,True,True,False'.split(',')

Given_data = {'level':level, 'lang':lang, 'tweets':tweets, 'phd':phd, 'hire':hire}
dataset = pd.DataFrame(Given_data,columns=['level','lang','tweets','phd','hire'])

"""
save the dataframe into a .csv file
"""

dataset.to_csv('C:/Users/ss3727s/OneDrive - Missouri State University/635 Data Mining/Assignment2/Assignment 2/hire_dataset.csv', index=False)

"""
load the hire.csv file
"""
df = pd.read_csv("C:/Users/ss3727s/OneDrive - Missouri State University/635 Data Mining/Assignment2/Assignment 2/hire_dataset.csv")


t = df.keys()[-1] #get the name of the target attribute

attribute_names = list(df.keys()) # Get the feature names from df


attribute_names.remove(t) #Remove the target attribute from the attribute names list


#Function to calculate the entropy of the whole dataset

def entropy(probs):  
    # return sum( [-prob*math.log(prob, 2) for prob in probs])
    return sum( [-prob*np.log2(prob) for prob in probs])

#Function to calulate probability
def probability_cal(ls,value):  
    
    total_instances = len(ls) # Total intances associated with respective attribute
    cnt = Counter(x for x in ls) #calculates the proportion
    probs = [x / total_instances for x in cnt.values()]  #probability equation
    return entropy(probs)

#function to calculate information gain

def information_gain(df, split_attribute, target_attribute,battr):
    
    df_split = df.groupby(split_attribute) # group based on attribute values
    
    glist=[]
    for gname,group in df_split:
        glist.append(gname) 
    
    glist.reverse()
    nobs = len(df.index) * 1.0   
    df_agg1=df_split.agg({target_attribute:lambda x:probability_cal(x, glist.pop())})
    df_agg2=df_split.agg({target_attribute :lambda x:len(x)/nobs})
    
    df_agg1.columns=['Entropy']
    df_agg2.columns=['Proportion']
    
    
    # Calculate Information Gain:
    new_entropy = sum( df_agg1['Entropy'] * df_agg2['Proportion'])
    
    if battr !='S':
        old_entropy = probability_cal(df[target_attribute],'S-'+df.iloc[0][df.columns.get_loc(battr)])
    else:
        old_entropy = probability_cal(df[target_attribute],battr)
    return old_entropy - new_entropy


#ID3 algorithm

def id3(df, target_attribute, attribute_names, default_class=None,default_attr='S'):
    
    cnt = Counter(x for x in df[target_attribute]) #counts the proportion of target attribute
    
    
    #If all target_values have the same value
    if len(np.unique(df[target_attribute])) <= 1:
        return np.unique(df[target_attribute])[0]
    
    #If the dataset is empty
    elif len(df)==0:
        return np.unique(df[target_attribute])[np.argmax(np.unique(df[target_attribute],return_counts=True)[1])]
    

    elif len(attribute_names) ==0:
        return default_class
    
    else:
        # Get Default Value for next recursive call of this function:
        default_class = max(cnt.keys())
        # default_class = cnt.most_common(1)[0][0]
        
        #Information Gain of the attributes:
        gainz=[]
        for attr in attribute_names:
            ig= information_gain(df, attr, target_attribute,default_attr)
            gainz.append(ig)

        index_of_max = gainz.index(max(gainz))               
        best_attr = attribute_names[index_of_max]            
       
        # Create an empty tree
        tree = {best_attr:{}} # Initiate the tree with best attribute as a node 
        remaining_attribute_names =[i for i in attribute_names if i != best_attr]
        
        
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target_attribute, remaining_attribute_names,default_class,best_attr)
            tree[best_attr][attr_val] = subtree
            # if attr_val == 'Junior':
            #     print(df)
            #     print(cnt)
            
        # if best_attr == 'Senior':
        #         print(df)
        #         print(cnt)
        #         print(cnt.most_common(1)[0][0])
            
        tree[best_attr][None] = cnt.most_common(1)[0][0]
        return tree



tree = id3(df,t,attribute_names)
print("\nThe Resultant Decision Tree of given dataset: \n")
pprint(tree)

def classify (root, sample):
    node = root
    
    while isinstance(node, dict):
        # print (node)
    
        att = list(node.keys())[0]
        val = sample.get(att, None)
        
        # print(att,val)       
        node = node[att]
        if node.get(val, None) == None:
            val = None
            
        node = node[val]
    return node

print ("\nThe Testing Cases for given dataset are: \n")       
sample1 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"}
sample2 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"}
sample3 = {"level" : "Intern"}
sample4 = {"level" : "Senior"}    

r1= classify(tree, sample1)
print (str(sample1),'=',r1)

r2= classify(tree, sample2)
print (str(sample2),'=',r2)

r3= classify(tree, sample3)
print (str(sample3),'=',r3)

r4= classify(tree, sample4)
print (str(sample4),'=',r4)

"""
Part 2 of Assignment2: Implementation of ID3 by your own dataset
"""

print ("--------------------------------------------------------------------------------------------------------------------------")

ds = pd.read_csv("C:/Users/ss3727s/OneDrive - Missouri State University/635 Data Mining/Assignment2/Assignment 2/play_tennis.csv")

testing_data =ds.drop('day', axis=1)


tar_test = testing_data.keys()[-1]

# Get the attribute names from input dataset
attribute_names_test = list(testing_data.keys())

#Remove the target attribute from the attribute names list
attribute_names_test.remove(tar_test) 


tree1 = id3(testing_data,tar_test,attribute_names_test)
print("\nThe Resultant Decision Tree of 'Play_Tennis.csv' is: \n")
pprint(tree1)


print ("\nThe Testing Cases are: \n")       
sample_t1 = {"outlook" : "Sunny","temp" : "Mild","humidity" : "High","wind" : "Weak"}
sample_t2 = {"outlook" : "Rain","temp" : "Cool","humidity" : "High","wind" : "Weak"}
sample_t3 = {"outlook" : "Snow"}
sample_t4 = {"outlook": "Sunny"}
   

rt1= classify(tree1, sample_t1)
print (str(sample_t1),'=',rt1)

rt2= classify(tree1, sample_t2)
print (str(sample_t2),'=',rt2)

rt3= classify(tree1, sample_t3)
print (str(sample_t3),'=',rt3)

rt4= classify(tree1, sample_t4)
print (str(sample_t4),'=',rt4)

