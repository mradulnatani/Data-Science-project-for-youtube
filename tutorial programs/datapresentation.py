#Learn about our potential customers. That is, learn the characteristics of those who choose to bank with us, as well as those who do not.
#Develop a profitable method of identifying likely positive responders, so that we may save time and money. That is, develop a model or models that will identify likely positive responders. Quantify the expected profit from using these models.
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
bank_train = pd.read_csv(r"D:\GIT\pythondev\1\bank.csv")
print(bank_train.shape)

bank_train['index'] = pd.Series(range(0,4521))
bank_train.head()
print(bank_train)
bank_train['previous'] = bank_train['previous'].replace({999: np.nan})
bank_train['previous'].plot(kind = 'hist',title = 'Histogram of Days Since Previous')
bank_train['education_numeric'] = bank_train['education']
print(bank_train)
dict_edu = {"education_numeric": {"primary": 0, 
"secondary": 1, "tertiary": 2,"unknown": np.nan}}
bank_train['education_numeric'] = bank_train['education_numeric'].replace(dict_edu["education_numeric"])
bank_train.replace(dict_edu, inplace=True)
bank_train['age_z'] = stats.zscore(bank_train['age'])
bank_train_outliers = bank_train.query('age_z > 3 | age_z < - 3')
bank_train_sort = bank_train.sort_values(['age_z'], ascending = False)
bank_train_sort[['age', 'marital']].head(n=15)
print(bank_train)
plt.show()



def create_bar_graphs(bank_train):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Graph for Previous Outcome
    crosstab_poutcome = pd.crosstab(bank_train['poutcome'], bank_train['y'])
    
    # Non-normalized bar graph
    crosstab_poutcome.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Bar Graph of Previous Outcome with Response Overlay')
    ax1.set_xlabel('Previous Outcome')
    ax1.set_ylabel('Count')
    
    # Normalized bar graph
    crosstab_poutcome_norm = crosstab_poutcome.div(crosstab_poutcome.sum(1), axis=0)
    crosstab_poutcome_norm.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Normalized Bar Graph of Previous Outcome with Response Overlay')
    ax2.set_xlabel('Previous Outcome')
    ax2.set_ylabel('Proportion')
    
    plt.tight_layout()
    plt.show()

    # Histogram for Age
def create_age_graphs(bank_train):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Separate age by response
    age_yes = bank_train[bank_train['y'] == 'yes']['age_z']
    age_no = bank_train[bank_train['y'] == 'no']['age_z']
    
    # Non-normalized histogram
    plt.sca(ax1)
    plt.hist([age_yes, age_no], bins=20, stacked=True, label=['Yes', 'No'])
    plt.title('Histogram of Age with Response Overlay')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Normalized histogram
    plt.sca(ax2)
    # Calculate normalized histogram
    h_yes, bin_edges = np.histogram(age_yes, bins=20)
    h_no, _ = np.histogram(age_no, bins=bin_edges)
    
    # Calculate proportions
    h_total = h_yes + h_no
    prop_yes = h_yes / h_total
    prop_no = h_no / h_total
    
    # Plot stacked proportional histogram
    plt.bar(bin_edges[:-1], prop_yes, width=np.diff(bin_edges), align='edge', alpha=0.5, label='Yes')
    plt.bar(bin_edges[:-1], prop_no, width=np.diff(bin_edges), align='edge', alpha=0.5, bottom=prop_yes, label='No')
    
    plt.title('Normalized Histogram of Age with Response Overlay')
    plt.xlabel('Age')
    plt.ylabel('Proportion')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Binning function
def perform_age_binning(bank_train):
    # Bin the age variable
    bank_train['age_binned'] = pd.cut(
        x=bank_train['age_z'], 
        bins=[0, 30, 50, 100],
        labels=["Young (0-30)", "Middle (31-50)", "Senior (51+)"]
    )
    
    # Create contingency table
    contingency_table = pd.crosstab(bank_train['age_binned'], bank_train['y'])
    
    # Create bar graph
    plt.figure(figsize=(10, 6))
    contingency_table.plot(kind='bar', stacked=True)
    plt.title('Response by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Response')
    plt.tight_layout()
    plt.show()
    
    # Print percentages
    print("Percentage of Response by Age Group:")
    percentage_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    print(percentage_table.round(2))

# Main execution
def main():
    print("1. Bar Graphs with Previous Outcome:")
    create_bar_graphs(bank_train)
    
    print("\n2. Age Histograms:")
    create_age_graphs(bank_train)
    
    print("\n3. Age Binning Analysis:")
    perform_age_binning(bank_train)

# Execute the main function
if __name__ == "__main__":
    main()