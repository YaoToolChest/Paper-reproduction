from transformers import AutoTokenizer,AutoModelForCausalLM,GPTNeoXForCausalLM,GPTNeoXTokenizerFast
import pandas as pd 
import torch



whether = True
###################加载csv
df = pd.read_csv('output_V3.csv')
PeoPle_Name=df['People_Name']
Birth_Data = df['Birth_Date']
Birth_City_Name = df['Birth_City_Name']
University_Name = df['University_Name']
Company_Name =df['Company_Name']
Major_Name =df['Major_Name']
Work_City_Name = df['Work_City_Name']
model_name = 'check_point_36'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


#############匹配函数
def contains_string(substring, main_string):
    """
    检查main_string中是否包含substring。

    参数:
    substring (str): 子字符串，需要在main_string中查找的字符串。
    main_string (str): 主字符串，将被搜索以查找substring。

    返回:
    bool: 如果main_string包含substring，则返回True；否则返回False。
    """
    return substring in main_string





def RightOrNot(Q,FeatureName,Feature,Total_Number,True_Number):
    
    Q_ids = tokenizer(Q, return_tensors="pt").input_ids
    ReA = model.generate(
    Q_ids,
    do_sample=False,
    max_length=128,
)  
    FeatureName_Token =tokenizer(Feature, return_tensors="pt").input_ids
                                                                                                                
    ReQA = tokenizer.decode(ReA[0])
    if whether :
        print("属性为"+FeatureName)
        print("正确答案为："+Feature)
        print("实际回答为："+ ReQA)
        print(FeatureName_Token)
        print(Q_ids[0])

    Total_Number=Total_Number + 1
    if contains_string(Feature, ReQA):
        True_Number=True_Number+1
        print("Ture")
    else:
        print('False')
    return Total_Number,True_Number
    
##########



City_number = 0
City_Total = 0

University_Number = 0
University_Total = 0

Date_number = 0
Date_Total = 0

True_Number=0
Total_Number=0
###################



for i in range(0,100000,1):
    PeopleName = PeoPle_Name[i]
    BirthData = Birth_Data[i]
    BirthCityName = Birth_City_Name[i]
    UniversityName = University_Name[i]
    MajorName = Major_Name[i]
    CompanyName = Company_Name[i]
    WorkCityName = Work_City_Name[i]
    
    Birth_DataQ='what is the birth data of  '+PeopleName+' ?'
    Birth_CityQ='what is the birth city of '+PeopleName+' ?'
    UniversityQ = 'Which university did '+PeopleName+' study ?'
    MajorQ = 'What major did '+PeopleName+' study ?'
    CompanyQ = 'which company did '+PeopleName+' work for'
    Work_CityQ = 'Where did '+PeopleName+' work ?'
    
    Total_Number,True_Number=RightOrNot(Birth_CityQ,'Birth_City',BirthCityName,Total_Number,True_Number)
    Total_Number,True_Number=RightOrNot(Birth_DataQ,'Birth_dataQ',BirthData,Total_Number,True_Number)
    Total_Number,True_Number=RightOrNot(UniversityQ,'University_Name',UniversityName,Total_Number,True_Number)
    Total_Number,True_Number=RightOrNot(MajorQ,'MajorName',MajorName,Total_Number,True_Number)
    Total_Number,True_Number=RightOrNot(CompanyQ,'CompanyName',CompanyName,Total_Number,True_Number)
    Total_Number,True_Number=RightOrNot(Work_CityQ,'WorkCityName',WorkCityName,Total_Number,True_Number)
    
    if Total_Number>100:
        break

print("===================================")
print(True_Number)
print(Total_Number)
print(True_Number/Total_Number)
