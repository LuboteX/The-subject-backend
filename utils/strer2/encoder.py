import json
import copy
def encoder(input_file_path,file_template_path,output_file_path):
    with open(input_file_path,'r',encoding="utf-8") as file:
        input_data = json.load(file)

    with open(file_template_path,'r',encoding="utf-8") as file:
        template_data = json.load(file)

    output_data = []

    for item in input_data:
        for personality_key in template_data['input']['personality']:
            if personality_key in input_data[item]:
                template_data['input']['personality'][personality_key] = input_data[item][personality_key]
        for input_key in template_data['input']:
            if input_key in input_data[item]:
                template_data['input'][input_key] = input_data[item][input_key]
        for output_key in template_data['output']:
            if output_key in input_data[item]:
                template_data['output'][output_key] = input_data[item][output_key]
        output_data.append(copy.deepcopy(template_data))

    print(output_data)

    #转换为json
    with open(output_file_path,'w',encoding="utf-8" ) as file:
        json.dump(output_data,file,ensure_ascii=False,indent=4)