import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# load data
url_train = "train.csv"
train = pd. read_csv (url_train)

train.describe ()
train["Lead"]. replace("Female", 1, inplace = True )
train["Lead"]. replace ("Male", 0, inplace = True )

Y = train["Lead"]
train["Lead"] = train["Lead"].astype("category")
# Q1: Do men or women dominate speaking roles in Hollywood movies?
print (f"female : {Y.sum ()/len (Y)* 100}%") # female is 1 and male is 0
print (f"male : {(len(Y)-Y.sum ())/ len(Y)*100}%")  # female is 1 and male is 0

# Q2: changed over time?
df = train.groupby(['Year',"Lead"])["Lead"].count()
df = pd.DataFrame(df)
df.to_csv("year.csv")

df = pd. read_csv ("year.csv")
df["Lead"] = df["Lead"].astype("category")

plt.figure(figsize=(16,10))
sns.set_style("white")
sns.lineplot(
    x="Year", y="Number",
    hue="Lead",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)

plt.show()

# question3: Do films in which men do more speaking make a lot more money than films in which women speak more?
gross = train.groupby(["Lead"]).agg({"Gross":['mean','median', 'max',"std"]})
gross
