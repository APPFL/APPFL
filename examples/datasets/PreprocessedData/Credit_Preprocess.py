import pandas as pd
import numpy as np
import scipy.stats as stats 

def text_cleaning(data):
    if data is np.NaN or not isinstance(data, str):
        return data
    else:
        return str(data).strip('_ ,"')

def Month_Converter(x):
    if pd.notnull(x):
        num1 = int(x.split(' ')[0])
        num2 = int(x.split(' ')[3])
      
        return (num1*12)+num2
    else:
        return x



def get_Diff_Values_Colum(df_column, diff_value=[], sep=',', replace=''):   
    column = df_column.dropna()
    for i in column:
        if sep not in i and i not in diff_value:
            diff_value.append(i)
        else:
            for data in map(lambda x:x.strip(), re.sub(replace, '', i).split(sep)):
                if not data in diff_value:
                    diff_value.append(data)
    return dict(enumerate(sorted(diff_value)))

# Reassign and Show Function
def Object_NaN_Values_Reassign_Group_Mode(df, groupby, column, inplace=True):      
    import numpy as np
    # Assigning Wrong values Make Simple Function
    def make_NaN_and_fill_mode(df, groupby, column, inplace=True):
        # Assign None to np.NaN
        if df[column].isin([None]).sum():
            df[column][df[column].isin([None])] = np.NaN
            
        # fill with local mode using np.unique
        result = df.groupby(groupby)[column].transform(lambda x: fillna_mode(x))

        # inplace
        if inplace:
            df[column] = result
        else:
            return result
    
    def fillna_mode(series):
        if series.dtype == 'float64':
            return series.fillna(stats.mode(series.dropna())[0][0])
        elif series.dtype == 'object':
            return series.fillna(series.mode().iloc[0])
        else:
            return series

    # Run      
    if inplace:  
        # Before Assigning NaN values   
        if df[column].value_counts(dropna=False).index.isna().sum():
            x = df[column].value_counts(dropna=False).loc[[np.NaN]]
            
        a = df.groupby(groupby)[column].apply(list) 
        
        # Assigning
        make_NaN_and_fill_mode(df, groupby, column, inplace)
        
        # After Assigning NaN values
        if df[column].value_counts(dropna=False).index.isna().sum():
            y = df[column].value_counts(dropna=False).loc[[np.NaN]]
            
        b = df.groupby(groupby)[column].apply(list)
    else:   
        # Show
        return make_NaN_and_fill_mode(df, groupby, column, inplace)

# Define Outlier Range
def get_iqr_lower_upper(df, column, multiply=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 -q1
    
    lower = q1-iqr*multiply
    upper = q3+iqr*multiply
    affect = df.loc[(df[column]<lower)|(df[column]>upper)].shape
    print('Outliers:', affect)
    return lower, upper

# Reassign Wrong Values and Show Function
def Numeric_Wrong_Values_Reassign_Group_Min_Max(df, groupby, column, inplace=True):      

    # Identify Wrong values Range
    def get_group_min_max(df, groupby, column):            
        cur = df[df[column].notna()].groupby(groupby)[column].apply(list)
        x, y = cur.apply(lambda x: stats.mode(x, axis=None, keepdims=True)).apply([min, max])
        return x[0], y[0]

    # Assigning Wrong values
    def make_group_NaN_and_fill_mode(df, groupby, column, inplace=True):
        df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
        x, y = df_dropped.apply(lambda x: stats.mode(x, axis=None, keepdims=True)).apply([min, max])
        mini, maxi = x[0], y[0]

        # assign Wrong Values to NaN
        col = df[column].apply(lambda x: np.NaN if ((x<mini)|(x>maxi)) else x)

        # fill with local mode
        mode_by_group = df.groupby(groupby)[column].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.NaN)
        result = col.fillna(mode_by_group)

        # inplace
        if inplace:
            df[column] = result
        else:
            return result
        
    
    # Run      
    if inplace:   
        # Before Assigning NaN values   
        if df[column].value_counts(dropna=False).index.isna().sum():
            x = df[column].value_counts(dropna=False).loc[[np.NaN]]
            
        mini, maxi = get_group_min_max(df, groupby, column)        
        
        a = df.groupby(groupby)[column].apply(list) 
        
        # Assigning
        make_group_NaN_and_fill_mode(df, groupby, column, inplace)
        
        # After Assigning NaN values
        if df[column].value_counts(dropna=False).index.isna().sum():
            y = df[column].value_counts(dropna=False).loc[[np.NaN]]
        
        b = df.groupby(groupby)[column].apply(list)
    else:   
        # Show
        return make_group_NaN_and_fill_mode(df, groupby, column, inplace)

class CreditData():
    def __init__(self):
        dir = "./datasets/RawData/Credit/archive/"
        self.train_data = pd.read_csv(dir+'train.csv',low_memory=False)

    def preprocess(self):
        print('Preprocessing Credit Data')
        df_copy1 = self.train_data
        df = df_copy1.applymap(text_cleaning).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN)

        df['ID']                      = df.ID.apply(lambda x: int(x, 16))
        df['Customer_ID']             = df.Customer_ID.apply(lambda x: int(x[4:], 16))
        df['Month']                   = pd.to_datetime(df.Month, format='%B').dt.month
        df['Age']                     = df.Age.astype(int) 
        df['SSN']                     = df.SSN.apply(lambda x: x if x is np.NaN else int(str(x).replace('-', ''))).astype(float)
        df['Annual_Income']           = df.Annual_Income.astype(float)
        df['Num_of_Loan']             = df.Num_of_Loan.astype(int) 
        df['Num_of_Delayed_Payment']  = df.Num_of_Delayed_Payment.astype(float)
        df['Changed_Credit_Limit']    = df.Changed_Credit_Limit.astype(float)
        df['Outstanding_Debt']        = df.Outstanding_Debt.astype(float)
        df['Amount_invested_monthly'] = df.Amount_invested_monthly.astype(float)
        df['Monthly_Balance']         = df.Monthly_Balance.astype(float)

        df['Credit_History_Age'] = df.Credit_History_Age.apply(lambda x: Month_Converter(x)).astype(float)
        df['Type_of_Loan'] = df['Type_of_Loan'].apply(lambda x: x.lower().replace('and ', '').replace(', ', ',').strip() if pd.notna(x) else x)
        
        Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Name')
        Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Occupation')
        
        df.groupby('Customer_ID')['Type_of_Loan'].value_counts(dropna=False)
        df['Type_of_Loan'].replace([np.NaN], 'No Data', inplace=True)
        Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Credit_Mix')
        Object_NaN_Values_Reassign_Group_Mode(df, 'Customer_ID', 'Payment_Behaviour')

        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Age')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'SSN')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Annual_Income')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Inhand_Salary')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Bank_Accounts')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Credit_Card')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Interest_Rate')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Loan')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Delay_from_due_date')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Delayed_Payment')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Changed_Credit_Limit')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_Credit_Inquiries')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Outstanding_Debt')


        df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age']\
                                    .apply(lambda x: x.interpolate().bfill().ffill())\
                                    .reset_index(level=0, drop=True)

        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Total_EMI_per_month')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Amount_invested_monthly')
        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Balance')

        df.loc[df['Num_Bank_Accounts']<0, 'Num_Bank_Accounts'] = 0
        df.loc[df['Delay_from_due_date']<0, 'Delay_from_due_date'] = None

        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Delay_from_due_date')

        df.loc[df['Num_of_Delayed_Payment']<0, 'Num_of_Delayed_Payment'] = None

        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Num_of_Delayed_Payment')

        df.loc[df['Monthly_Balance']<0, 'Monthly_Balance'] = None

        Numeric_Wrong_Values_Reassign_Group_Min_Max(df, 'Customer_ID', 'Monthly_Balance')

        df.loc[df['Amount_invested_monthly']>=10000, 'Amount_invested_monthly'] = None
        df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.NaN)

        df.to_csv('./datasets/PreprocessedData/Credit_Preprocess.csv', index=False)
        print('Preprocessing Completed. File is saved as Credit_Preprocess.csv')
    