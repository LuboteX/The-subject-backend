import json
import csv

#读取原始数据
data = []
with open("output_csv.csv",encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(len(row))
        print(row)
        if None in row:
            del row[None]
            data.append(row)


#写入json
with open('new_output_json.json','w',encoding="utf-8" ) as file:
    json.dump(data,file,ensure_ascii=False)   