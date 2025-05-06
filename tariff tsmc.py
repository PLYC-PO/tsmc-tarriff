import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

#匯入估計期間個股資料並處理缺失值
df_tsmc_9 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202409.csv',encoding='big5',skiprows=1,skipfooter=6,sep=",")
df_tsmc_9.drop(columns=["Unnamed: 2"],inplace=True)

df_tsmc_10 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202410.csv', encoding='big5',skiprows=1,skipfooter=6,sep = ',',)
df_tsmc_10.drop(columns=["Unnamed: 2"],inplace=True)

df_tsmc_11 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202411.csv', encoding='big5',skiprows=1,skipfooter=6,sep = ',',)
df_tsmc_11.drop(columns=["Unnamed: 2"],inplace=True)

df_tsmc_12 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202412.csv', encoding='big5',skiprows=1,skipfooter=6,sep = ',',)
df_tsmc_12.drop(columns=["Unnamed: 2"],inplace=True)

temp1=pd.merge(df_tsmc_9,df_tsmc_10,on=['日期','收盤價'],how='outer')
temp2=pd.merge(df_tsmc_11,df_tsmc_12,on=['日期','收盤價'],how='outer')
#合併個股數據
df_tsmc=pd.merge(temp1,temp2,on=['日期','收盤價'],how='outer')
pd.set_option('display.max_rows', None)

#計算個股每個交易日間的報酬率
df_tsmc['收盤價'] = df_tsmc['收盤價'].str.replace(',', '').astype(float)
df_tsmc['報酬率'] =  df_tsmc['收盤價'].pct_change()
print(df_tsmc)



#匯入估計期間的加權指數數據
df_taiex_9= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST.csv',encoding='big5',skiprows=1,sep=",")
df_taiex_9.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)


df_taiex_10= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (1).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_10.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)


df_taiex_11= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (2).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_11.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)

df_taiex_12= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (3).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_12.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)


temp3=pd.merge(df_taiex_9,df_taiex_10,on=['日期','收盤指數'],how='outer')
temp4=pd.merge(df_taiex_11,df_taiex_12,on=['日期','收盤指數'],how='outer')
#合併個股數據
df_taiex=pd.merge(temp3,temp4,on=['日期','收盤指數'],how='outer')
pd.set_option('display.max_rows', None)

df_taiex['收盤指數'] = df_taiex['收盤指數'].str.replace(',', '').astype(float)
df_taiex['市場報酬率'] =  df_taiex['收盤指數'].pct_change()
df_taiex.dropna(subset=['市場報酬率'],inplace=True)
print(df_taiex)

df_model=pd.merge(df_tsmc,df_taiex,on=['日期'],how='outer')
print(df_model)
df_model.drop(index=0,inplace=True) #清掉第一天
print(df_model)

#跑回歸算截距
model=smf.ols(formula='報酬率~市場報酬率',data=df_model).fit()
print(model.summary())
alpha=model.params['Intercept']
beta=model.params['市場報酬率']
print(alpha,beta)


#不是估計也不是事件期間 就是為了做圖中間空白的數據,114年2,3月
#114年2,3月的個股
'''df_tsmc_2 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202502.csv',encoding='big5',skiprows=1,skipfooter=6,sep=",")
df_tsmc_2.drop(columns=["Unnamed: 2"],inplace=True)

df_tsmc_3 = pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202503.csv', encoding='big5',skiprows=1,skipfooter=6,sep = ',',)
df_tsmc_3.drop(columns=["Unnamed: 2"],inplace=True)

#114年2,3月的市場報酬
df_taiex_2= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (5).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_2.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)


df_taiex_3= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (1).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_3.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)'''



#事件期間的個股報酬率

df_tsmc_4=pd.read_csv('/Users/pl/Downloads/STOCK_DAY_AVG_2330_202504.csv',encoding='big5',skiprows=1,skipfooter=6,sep=",")
df_tsmc_4.drop(columns=["Unnamed: 2"],inplace=True)
df_tsmc_4=df_tsmc_4.head()

df_tsmc_4['報酬率']=df_tsmc_4['收盤價'].pct_change()
df_tsmc_4.drop(index=0,inplace=True)
print(df_tsmc_4)


#事件期間的市場報酬率
df_taiex_4= pd.read_csv('/Users/pl/Downloads/MI_5MINS_HIST (4).csv',encoding='big5',skiprows=1,sep=",")
df_taiex_4.drop(columns=['開盤指數', '最高指數', '最低指數','Unnamed: 5'],inplace=True)
df_taiex_4=df_taiex_4.head()
df_taiex_4['收盤指數']= df_taiex_4['收盤指數'].str.replace(',','').astype(float)
df_taiex_4['市場報酬率']= df_taiex_4['收盤指數'].pct_change()
df_taiex_4.drop(index=0,inplace=True)
print(df_taiex_4)

#合併事件期間的個股和市場數據

df_tariff=pd.merge(df_tsmc_4,df_taiex_4,on=['日期'],how='outer')
print(df_tariff)


#合併估計期間和事件期間
df_final=pd.merge(df_model,df_tariff,on=['日期','收盤價','報酬率','收盤指數','市場報酬率'],how='outer')
print(df_final)

#估計期間異常報酬
df_final['預期報酬率']= alpha + beta * df_final['市場報酬率']
df_final['異常報酬']= df_final['報酬率']- df_final['預期報酬率']
df_final['累積異常報酬']=df_final['異常報酬'].cumsum()
print(df_final)

#轉換日期格式
date_split=df_final['日期'].str.split('/',expand=True)
df_final['西元日期']=((date_split[0].astype(int)+1911).astype(str)+'/'+date_split[1]+'/'+date_split[2])
df_final['日期'] = pd.to_datetime(df_final['西元日期'], format="%Y/%m/%d")
print(df_final['日期'])
tariff_annouce=pd.to_datetime("2025-04-02")
df_final['日期']=(df_final['日期']-tariff_annouce).dt.days
print(df_final['日期'])


'''#利用seaborn 畫的圖
sns.scatterplot(x='日期',y='異常報酬',data=df_final)#這張圖是AR.....
plt.show()'''

sns.scatterplot(x='日期',y='累積異常報酬',data=df_final,)
plt.xlabel("Date")                  # ➜ 加上 X 軸標籤
plt.ylabel("Cumulative Abnormal Return")
plt.show()


