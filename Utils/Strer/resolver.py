import re
import configparser
import json
import csv

file_address = "utils/strer/input.txt"
config_address = "utils/strer/config.ini"

def split_str_by_mulsymbol(str,symbol):
    pattern = '[' + re.escape(symbol) + ']'
    return re.split(pattern=pattern,string= str)

#读取原始数据，分割处理
with open(file_address, encoding="utf-8") as f:
    doc = f.read()
doclist = split_str_by_mulsymbol(doc,"<>:")
doclist = list(filter(None,doclist))

#字段配置
config = configparser.ConfigParser()
config.read(config_address,encoding="utf-8")
fields = config.sections()
orderfieldDict = {}
for index in range(len(fields)):
    orderfieldDict[fields[index]] = index

#空内容字段字典
nullelem={}
for field in fields:
    nullelem[field] = ''

#最终数据
data=[]

#处理
elem=dict(nullelem)
for index in range(len(doclist)):
    if index%2 == 0:
        elem[doclist[index]]=doclist[index+1]
    #method1 按升序分割
    # if(index>=len(doclist)-2):
    #     data.append(elem)
    #     elem = dict(nullelem)
    #     break
    # if index%2 == 0 and orderfieldDict[doclist[index]] >= orderfieldDict[doclist[index+2]]:
    #     data.append(elem)
    #     elem = dict(nullelem)
        
    #method2 按最后一个字段分割
    if index%2 == 0 and orderfieldDict[doclist[index]] == len(fields)-1:
        data.append(elem)
        elem = dict(nullelem)

#转换为json
with open('utils/strer/output/output_json.json','w',encoding="utf-8" ) as file:
    json.dump(data,file,ensure_ascii=False)        

#转换为csv
with open("utils/strer/output/output_csv.csv",'w',encoding="utf-8",newline='') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=fields)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(data)

