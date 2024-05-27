    
import re
import os
from utils.strer2 import encoder
# 初始文件路径和内容
base_path = './document/training_data/raw_data/'
base_filename = 'raw_data_hate_'
file_extension = '.json'
content = 'response content here'  # 用实际内容替换此字符串
input_file_path = "document/training_data/raw_data/raw_data_hate_002.json"
file_template_path = "document/training_data/template.json"
output_base_path = "utils/strer2/output/encode_data_hate_"
# 获取现有文件列表并找出最大的编号
existing_files = [f for f in os.listdir(base_path) if f.startswith(base_filename) and f.endswith(file_extension)]
existing_numbers = [int(re.search(r'(\d{3})', f).group()) for f in existing_files]

if existing_numbers:
    start_number = max(existing_numbers) + 1
else:
    start_number = 1

# 定义要生成的文件数量
num_files = 4  # 可以根据需要修改

# 循环生成文件
for i in range(start_number, start_number + num_files):
    file_number = str(i).zfill(3)  # 生成三位数的编号，例如 001, 002
    filename = f'{base_filename}{file_number}{file_extension}'
    file_path = os.path.join(base_path, filename)
    
    with open(file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(content)
    output_file_path = os.path.join(output_base_path,file_number,file_extension)
    encoder.encoder(file_path,file_template_path,output_file_path)