import json
import os
from openai import OpenAI
import requests
import urllib3
import os
import requests
import json
import base64
from PIL import Image
import pickle
from io import BytesIO
import re


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_response_from_chatgpt_openai(prompt: str, model_name: str, temperature: str):
    os.environ["http_proxy"] = "http://10.105.20.64:7890"
    os.environ["https_proxy"] = "http://10.105.20.64:7890"
    # Set up the OpenAI API client

    client = OpenAI(
        ## wzl
        api_key = "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",
    )

    temperature_float = float(temperature)
    # Generate a response
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature_float
    )

    response = completion.choices[0].message.content.strip()

    return response


# def get_response_from_qwen(prompt: str):
#     url = "http://localhost:11434/api/chat"
#     headers = {'Content-Type': 'application/json'}
#     data = {
#         "model": "qwen2:7b",
#         "messages": [{"role": "user", "content": prompt}],
#         "stream": False
#         }
#
#     response = requests.post(url, json=data, headers=headers)
#
#     if response.status_code == 200:
#         return response.json()['message']['content']
#     else:
#         return {"error": "Failed to generate response", "status_code": response.status_code}
#
# def get_response_from_qwen(prompt: str,ent1,ent2_list):
#
#     # print(1242151235)
#     def create_blank_image(width=224, height=224, color=(255, 255, 255)):
#         """创建一个空白图片作为占位符"""
#         blank_image = Image.new('RGB', (width, height), color)
#         return blank_image
#
#     zh_images = pickle.load(open("../ent_img/ent_img/zh_dbp15k_link_img_dict_full.pkl", 'rb'))
#     en_images = pickle.load(open("../ent_img/ent_img/en(zh)_dbp15k_link_img_dict_full.pkl", 'rb'))
#
#     # 从图像数据中获取指定实体的图片
#     # print(type(ent2_list))
#     blank_image = create_blank_image()
#
#     ent2, ent3, ent4, ent5, ent6 = ent2_list
#     # 获取所有实体的图片，如果没有则使用空白图片
#     images = {
#         "Source Entity": zh_images.get(ent1, blank_image),
#         "Target Entity 1": en_images.get(ent2, blank_image),
#         "Target Entity 2": en_images.get(ent3, blank_image),
#         "Target Entity 3": en_images.get(ent4, blank_image),
#         "Target Entity 4": en_images.get(ent5, blank_image),
#         "Target Entity 5": en_images.get(ent6, blank_image)
#     }
#     # 检查哪些图片是真实的，哪些是空白占位符
#     missing_images = []
#     for name, img in images.items():
#         if img == blank_image:
#             missing_images.append(name)
#     # 准备图片数据
#     image_base64_list = []
#
#     # 将图片转换为base64
#     def image_to_bytes(img):
#         img_byte_arr = BytesIO()
#         img.save(img_byte_arr, format='PNG')
#         return img_byte_arr.getvalue()
#
#     # 按固定顺序添加图片 (Source, Target1-5)
#     image_order = [
#         ("Source Entity", images["Source Entity"]),
#         ("Target Entity 1", images["Target Entity 1"]),
#         ("Target Entity 2", images["Target Entity 2"]),
#         ("Target Entity 3", images["Target Entity 3"]),
#         ("Target Entity 4", images["Target Entity 4"]),
#         ("Target Entity 5", images["Target Entity 5"])
#     ]
#     for name, img in image_order:
#         img = img.resize((224, 224))
#         image_bytes = image_to_bytes(img)
#         image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#         image_base64_list.append(image_base64)
#
#
#     url = "http://localhost:11434/api/generate"
#     data = {
#         "model": "qwen2.5vl:7b",
#         "prompt": prompt,
#         "stream": False,
#         "images": image_base64_list
#     }
#
#     response = requests.post(url, json=data,stream=False)
#     # print(5398539)
#     # print(f"response: {response.text}")
#     # 解析完整的JSON响应
#     json_data = json.loads(response.text)
#     # print(f"response: {json_data}")
#     if "response" in json_data:
#         # print(json_data["response"])
#         return json_data["response"]
#
#     else:
#         return {"error": "Failed to generate response", "status_code": response.status_code}





# 在模块级别加载PKL文件（只需要加载一次）
print(1111111111)
ZH_IMAGES = pickle.load(open("/Data/ljm/HLMEA-main/HLMEA-main/entimg/zh_dbp15k_link_img_dict_full.pkl", 'rb'))
print(2222222333)
EN_IMAGES = pickle.load(open("/Data/ljm/HLMEA-main/HLMEA-main/entimg/en(zh)_dbp15k_link_img_dict_full.pkl", 'rb'))
# EN_IMAGES = pickle.load(open("/Data/ljm/HLMEA-main/HLMEA-main/entimg/en(zh)_dbp15k_link_img_dict_full.pkl", 'rb'))
print(333333343)

def get_response_from_qwen(prompt: str, ent1, ent2_list):

    def clean_url(text):
    # 定义正则表达式模式
        # print(111333444)
    # 匹配 [内容] 或者 <|begin_of_box|>内容<|end_of_box|>
    # (.*?) 是我们要提取的 URL 部分
        pattern = r"(?:\[|<\|begin_of_box\|>)(.*?)(?:\]|<\|end_of_box\|>)"
        
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        # print(text)
        return text  # 如果没有匹配到特定格式，返回原文本

    def create_blank_image(width=224, height=224, color=(255, 255, 255)):
        """创建一个空白图片作为占位符"""
        blank_image = Image.new('RGB', (width, height), color)
        return blank_image

    # 使用全局变量而不是每次加载PKL文件
    blank_image = create_blank_image()

    ent2, ent3, ent4, ent5, ent6,ent7, ent8, ent9, ent10, ent11 = ent2_list
    # 获取所有实体的图片，如果没有则使用空白图片
    images = {
        "Source Entity": ZH_IMAGES.get(ent1, blank_image),
        "Target Entity 1": EN_IMAGES.get(ent2, blank_image),
        "Target Entity 2": EN_IMAGES.get(ent3, blank_image),
        "Target Entity 3": EN_IMAGES.get(ent4, blank_image),
        "Target Entity 4": EN_IMAGES.get(ent5, blank_image),
        "Target Entity 5": EN_IMAGES.get(ent6, blank_image),
        "Target Entity 6": EN_IMAGES.get(ent7, blank_image),
        "Target Entity 7": EN_IMAGES.get(ent8, blank_image),
        "Target Entity 8": EN_IMAGES.get(ent9, blank_image),
        "Target Entity 9": EN_IMAGES.get(ent10, blank_image),
        "Target Entity 10": EN_IMAGES.get(ent11, blank_image)
    }

    # 检查哪些图片是真实的，哪些是空白占位符
    missing_images = []
    for name, img in images.items():
        if img == blank_image:
            missing_images.append(name)

    # 准备图片数据
    image_base64_list = []

    # 将图片转换为base64
    def image_to_bytes(img):
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    def pil_to_base64(pil_image):
        buffer = BytesIO()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")  # ✅ 转为 RGB
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 按固定顺序添加图片 (Source, Target1-5)
    image_order = [
        ("Source Entity", images["Source Entity"]),
        ("Target Entity 1", images["Target Entity 1"]),
        ("Target Entity 2", images["Target Entity 2"]),
        ("Target Entity 3", images["Target Entity 3"]),
        ("Target Entity 4", images["Target Entity 4"]),
        ("Target Entity 5", images["Target Entity 5"]),
        ("Target Entity 6", images["Target Entity 6"]),
        ("Target Entity 7", images["Target Entity 7"]),
        ("Target Entity 8", images["Target Entity 8"]),
        ("Target Entity 9", images["Target Entity 9"]),
        ("Target Entity 10", images["Target Entity 10"]),

    ]

    # for name, img in image_order:
    #     img = img.resize((224, 224))
    #     image_bytes = image_to_bytes(img)
    #     image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    #     image_base64_list.append(image_base64)
    content_list = []
    for name, img in image_order:
        img = img.resize((224, 224))
        image_bytes = pil_to_base64(img)
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_bytes}"
            }
        })
    # url = "http://localhost:11434/api/generate"
    # data = {
    #     "model": "qwen3-vl:32b？ ",
    #     "prompt": prompt,
    #     "stream": False,
    #     "images": image_base64_list
    # }
    #
    # response = requests.post(url, json=data, stream=False)
    # json_data = json.loads(response.text)
    content_list.append({
        "type": "text",
        "text": prompt
    })
    data = {
        "model": "glm",
        "chat_template_kwargs":{"enable_thinking":False},
        "messages": [

            {
                "role": "user",
                "content": content_list
            }
        ],
        "temperature": 0.0
    }
    API_BASE = "http://192.168.135.219:8000/v1"
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data, indent=2)
    )

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        # print(answer)
        formatted_answer = clean_url(answer)
        print(formatted_answer)
        return formatted_answer
    # if "response" in json_data:
    #     return json_data["response"]
    else:
        return {"error": "Failed to generate response", "status_code": response.status_code}

def get_qianfan_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=nVFbEFrf8AcamqS60ohsC9xL&client_secret=IqLvQ7o0y6sU98XSxrhalAmIrgf9SURL"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def get_response_from_ernie(prompt: str, temperature: str):
    access_token = '24.6949dde494d9236abaeba7d318e23cc4.2592000.1718269449.282335-70479836'
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k-0329?access_token=" + access_token

    if temperature == '0':
        temperature = float(0.01)
    else:
        temperature = float(temperature)
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json().get("result")
    else:
        return {"error": "Failed to generate response", "status_code": response.status_code}


def get_response_from_llm(type: str, prompt: str, model_name: str, temperature: str,ent1,ent2_list):
    if type in ['gpt3.5', 'gpt4']:
        response = get_response_from_chatgpt_openai(prompt, model_name, temperature)
    elif type == 'qwen':
        response = get_response_from_qwen(prompt,ent1,ent2_list)
    elif type == 'ernie':
        response = get_response_from_ernie(prompt, temperature)
    else:
        raise RuntimeError('LLM name error: ' + type)

    return response


if __name__ == '__main__':
    # prompt = "You are a helpful assistant for the task of entity alignment (EA). Inputs are a source entity and a list of target entities. Your task is to select a most similar target entity regarding to the source entity in terms of semantics and graph structure. You reply with brief, to-the-point answers with no elaboration as truthfully as possible. An entity is described via a set of triples <subject, predicate, object>, for example <France, hasCapital, Paris> means that France has a capital city of Paris.\
    #         Here is an example:\
    #         [[source entity domain]]: [[http://dbpedia.org]]\
    #         [[source entity IRI]]: [[http://dbpedia.org/resource/Ralf_Loose]]\
    #         [[triples]]: [[(Ralf_Loose, birthDate, ""1963-01-05""^^<http://www.w3.org/2001/XMLSchema#date>), (Ralf_Loose, height, ""1.88""^^<http://www.w3.org/2001/XMLSchema#double>), (Ralf_Loose, team, Rot-Weiß_Oberhausen), (Ralf_Loose, birthPlace, Dortmund), (Ralf_Loose, managerClub, Dynamo_Dresden)]]\
    #         [[target entity domain]]: [[http://de.dbpedia.org]]\
    #         [[target entity1 IRI]]: [[http://de.dbpedia.org/resource/Ralf_Loose]]\
    #         [[triples]]: [[(Ralf_Loose, birthDate, ""1963-01-05""^^<http://www.w3.org/2001/XMLSchema#date>), (Ralf_Loose, height, ""1.88""^^<http://www.w3.org/2001/XMLSchema#double>), (Ralf_Loose, years, ""1986""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Ralf_Loose, appearancesInLeague, ""188""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Ralf_Loose, nationalYears, ""1977""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Ralf_Loose, club, Rot-Weiß_Oberhausen), (Ralf_Loose, birthPlace, Dortmund), (Ralf_Loose, trainerClub, 1._FSV_Mainz_05), (Ralf_Loose, managerClub, Dynamo_Dresden)]]\
    #         [[target entity2 IRI]]: [[http://de.dbpedia.org/resource/Ralf_Sträßer]]\
    #         [[triples]]: [[(Ralf_Sträßer, birthDate, ""1958-06-20""^^<http://www.w3.org/2001/XMLSchema#date>), (Ralf_Sträßer, years, ""1987""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Ralf_Sträßer, appearancesInLeague, ""79""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Ralf_Sträßer, nationalYears, ""1982""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Ralf_Sträßer, appearancesInNationalTeam, ""4""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Ralf_Sträßer, club, 1._FC_Union_Berlin), (Ralf_Sträßer, youthClub, BFC_Dynamo)]]\
    #         [[target entity3 IRI]]: [[http://de.dbpedia.org/resource/Ralf_Sturm]]\
    #         [[triples]]: [[(Ralf_Sturm, birthDate, ""1968-10-18""^^<http://www.w3.org/2001/XMLSchema#date>), (Ralf_Sturm, height, ""1.8""^^<http://www.w3.org/2001/XMLSchema#double>), (Ralf_Sturm, years, ""1994""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Ralf_Sturm, appearancesInLeague, ""63""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Ralf_Sturm, appearancesInNationalTeam, ""4""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Ralf_Sturm, club, Rot-Weiß_Oberhausen)]]\
    #         [[target entity4 IRI]]: [[http://de.dbpedia.org/resource/Rudolf_Rahn]]\
    #         [[triples]]: [[(Rudolf_Rahn, birthDate, ""1900-03-16""^^<http://www.w3.org/2001/XMLSchema#date>), (Rudolf_Rahn, viafId, 57560515), (Rudolf_Rahn, individualisedGnd, 124997155), (Rudolf_Rahn, lccn, n/90/635131), (Rudolf_Rahn, deathDate, ""1975-01-07""^^<http://www.w3.org/2001/XMLSchema#date>), (Rudolf_Rahn, birthPlace, Ulm), (Rudolf_Rahn, deathPlace, Düsseldorf)]]\
    #         [[target entity5 IRI]]: [[http://de.dbpedia.org/resource/Anthony_Radziwill]]\
    #         [[triples]]: [[(Anthony_Radziwill, birthDate, ""1959-08-04""^^<http://www.w3.org/2001/XMLSchema#date>), (Anthony_Radziwill, nick, Radziwiłł, Antoni Stanisław Albrecht (vollständiger Geburtsname)), (Anthony_Radziwill, viafId, 14192305), (Anthony_Radziwill, lccn, n/2006/13470), (Anthony_Radziwill, deathDate, ""1999-08-10""^^<http://www.w3.org/2001/XMLSchema#date>), (Anthony_Radziwill, birthPlace, Lausanne)]]\
    #         [[Answer]]: [[http://de.dbpedia.org/resource/Ralf_Loose]]\
    #         Here is the query:\
    #         [[source entity domain]]: [[http://dbpedia.org]]\
    #         [[source entity IRI]]: [[http://dbpedia.org/resource/Pablo_Cavallero]]\
    #         [[triples]]: [[(Pablo_Cavallero, birthDate, ""1974-04-13""^^<http://www.w3.org/2001/XMLSchema#date>), (Pablo_Cavallero, height, ""1.84""^^<http://www.w3.org/2001/XMLSchema#double>), (Pablo_Cavallero, team, Peñarol)]]\
    #         [[target entity domain]]: [[http://de.dbpedia.org]]\
    #         [[target entity1 IRI]]: [[http://de.dbpedia.org/resource/Pablo_Cavallero]]\
    #         [[triples]]: [[(Pablo_Cavallero, birthDate, ""1974-04-13""^^<http://www.w3.org/2001/XMLSchema#date>), (Pablo_Cavallero, nick, Cavallero Rodríguez, Pablo Oscar (vollständiger Name)), (Pablo_Cavallero, height, ""1.84""^^<http://www.w3.org/2001/XMLSchema#double>), (Pablo_Cavallero, years, ""2000""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Pablo_Cavallero, appearancesInLeague, ""26""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Pablo_Cavallero, club, Espanyol_Barcelona), (Pablo_Cavallero, trainerClub, Estudiantes_de_La_Plata), (Pablo_Cavallero, managerClub, CA_Independiente)]]\
    #         [[target entity2 IRI]]: [[http://de.dbpedia.org/resource/Matías_Fritzler]]\
    #         [[triples]]: [[(Matías_Fritzler, birthDate, ""1986-08-23""^^<http://www.w3.org/2001/XMLSchema#date>), (Matías_Fritzler, nick, Fritzler, Matías Lionel (vollständiger Name)), (Matías_Fritzler, height, ""1.79""^^<http://www.w3.org/2001/XMLSchema#double>), (Matías_Fritzler, years, ""2015""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Matías_Fritzler, appearancesInLeague, ""9""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Matías_Fritzler, club, Hércules_Alicante)]]\
    #         [[target entity3 IRI]]: [[http://de.dbpedia.org/resource/Pablo_Zabaleta]]\
    #         [[triples]]: [[(Pablo_Zabaleta, birthDate, ""1985-01-16""^^<http://www.w3.org/2001/XMLSchema#date>), (Pablo_Zabaleta, nick, Zabaleta, Pablo Javier (vollständiger Name)), (Pablo_Zabaleta, height, ""1.76""^^<http://www.w3.org/2001/XMLSchema#double>), (Pablo_Zabaleta, years, ""2008""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Pablo_Zabaleta, appearancesInLeague, ""80""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Pablo_Zabaleta, club, Espanyol_Barcelona), (Pablo_Zabaleta, birthPlace, Buenos_Aires), (Pablo_Zabaleta, youthClub, CA_San_Lorenzo_de_Almagro)]]\
    #         [[target entity4 IRI]]: [[http://de.dbpedia.org/resource/Diego_Calvo]]\
    #         [[triples]]: [[(Diego_Calvo, birthDate, ""1991-03-25""^^<http://www.w3.org/2001/XMLSchema#date>), (Diego_Calvo, nick, Calvo Fonseca, Diego Gerardo (vollständiger Name)), (Diego_Calvo, height, ""1.78""^^<http://www.w3.org/2001/XMLSchema#double>), (Diego_Calvo, years, ""2014""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Diego_Calvo, appearancesInLeague, ""34""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Diego_Calvo, club, IFK_Göteborg), (Diego_Calvo, birthPlace, Costa_Rica)]]\
    #         [[target entity5 IRI]]: [[http://de.dbpedia.org/resource/Pablo_Contreras]]\
    #         [[triples]]: [[(Pablo_Contreras, birthDate, ""1978-09-11""^^<http://www.w3.org/2001/XMLSchema#date>), (Pablo_Contreras, nick, Contreras Fica, Pablo Andrés (vollständiger Name)), (Pablo_Contreras, height, ""1.81""^^<http://www.w3.org/2001/XMLSchema#double>), (Pablo_Contreras, years, ""2012""^^<http://www.w3.org/2001/XMLSchema#gYear>), (Pablo_Contreras, appearancesInLeague, ""84""^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>), (Pablo_Contreras, club, Sporting_Braga), (Pablo_Contreras, nationalTeam, Chilenische_Fußballnationalmannschaft)]]\
    #         Which target entity do you think is the most similar one regarding to the source entity? Please answer with exactly one target entity IRI without any description."

    prompt = "Hello, 你好"
    llm_type = 'qwen'
    llm_model = 'gpt-3.5-turbo-1106'
    llm_temp = '0'
    message = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
    print(message)
