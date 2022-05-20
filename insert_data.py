from transformers import RoFormerModel, RoFormerTokenizer
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import pymysql
import time
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dunum =[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def title_insert():
    mysqlconnection = pymysql.connect(host='10.1.0.177',
                                     user='bwj',
                                     password='bwj',
                                     database='patent',
                                     cursorclass=pymysql.cursors.DictCursor)
    results = []
    with mysqlconnection:
        with mysqlconnection.cursor() as cursor:
            # Read a single record
            sql = "SELECT  `id`, `title` FROM `patent`"
            cursor.execute(sql)
            results = cursor.fetchall()
    # text = "所述荧光探针是以NaYF4、NaGdF4、CaF2、LiYF4、NaLuF4、LiLuF4、KMnF3或Y2O3为发光基质"
    connections.connect(host='localhost', port='19530')
    collection = Collection("patent")
    tokenizer = RoFormerTokenizer.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("./roformer_chinese_sim_char_ft_base")
    id_list = []
    embedding_list = []
    for result in results:
        id = result["id"]
        title = result["title"]
        pt_inputs = tokenizer(title, max_length=64, padding=True, return_tensors="pt")
        pt_outputs = pt_model(**pt_inputs)
        # print(pt_outputs["last_hidden_state"][0][0])
        embedding = pt_outputs["last_hidden_state"][0][0].tolist()
        id_list.append(id)
        embedding_list.append(embedding)

    # Get an existing collection.
    data = [id_list, embedding_list ]
    connections.connect(host='localhost', port='19530')
    collection = Collection("patent")
    mr = collection.insert(data)
    print(mr)

def signory_insert():
    mysqlconnection = pymysql.connect(host='10.1.0.177',
                                     user='root',
                                     password='root',
                                     database='patent',
                                     cursorclass=pymysql.cursors.DictCursor)
    results = []
    with mysqlconnection:
        with mysqlconnection.cursor() as cursor:
            # Read a single record
            sql = "SELECT  `signory_id`, `patent_id`, `signory_seg` FROM `signory_seg`"
            cursor.execute(sql)
            results = cursor.fetchall()
    connections.connect(host='localhost', port='19530')
    collection = Collection("signory")
    tokenizer = RoFormerTokenizer.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = torch.nn.DataParallel(pt_model, device_ids=dunum)
    pt_model.to(device)
    patent_id_list = []
    signory_id_list = []
    embedding_list = []
    time_start = time.time()
    for result in results:
        if len(patent_id_list) >= 1000:
            # Get an existing collection.
            data = [signory_id_list, patent_id_list, embedding_list]
            # 需要插入时解注释，防止误插入
            mr = collection.insert(data)
            print(mr)
            patent_id_list = []
            signory_id_list = []
            embedding_list = []
        signory_id = result["signory_id"]
        patent_id = result["patent_id"]
        signory = result["signory_seg"]
        print("signory_id:" + str(signory_id))
        pt_inputs = tokenizer(signory, return_tensors="pt")
        pt_outputs = pt_model(**pt_inputs)
        # print(pt_outputs["last_hidden_state"][0][0])
        embedding = pt_outputs["last_hidden_state"][0][0].tolist()
        patent_id_list.append(patent_id)
        signory_id_list.append(signory_id)
        embedding_list.append(embedding)
        time_now = time.time()
        time_past = time_now - time_start
        print(time_past)
    data = [signory_id_list, patent_id_list, embedding_list]
    # 需要插入时解注释，防止误插入
    mr = collection.insert(data)
    print(mr)

def abstract_insert():
    mysqlconnection = pymysql.connect(host='10.1.0.177',
                                     user='root',
                                     password='root',
                                     database='patent',
                                     cursorclass=pymysql.cursors.DictCursor)
    results = []
    with mysqlconnection:
        with mysqlconnection.cursor() as cursor:
            # Read a single record
            sql = "SELECT  `id`, `abstract` FROM `patent`"
            cursor.execute(sql)
            results = cursor.fetchall()
    connections.connect(host='localhost', port='19530')
    collection = Collection("abstract")
    tokenizer = RoFormerTokenizer.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = torch.nn.DataParallel(pt_model, device_ids=dunum)
    pt_model.to(device)
    id_list = []
    embedding_list = []
    time_start = time.time()
    for result in results:
        if len(id_list) >= 1000:
            # Get an existing collection.
            data = [id_list, embedding_list]
            # 需要插入时解注释，防止误插入
            mr = collection.insert(data)
            print(mr)
            id_list = []
            embedding_list = []
        id = result["id"]
        abstract = result["abstract"]
        print("id:" + str(id))
        pt_inputs = tokenizer(abstract, truncation=True, max_length=505, return_tensors="pt")
        pt_outputs = pt_model(**pt_inputs)
        # print(pt_outputs["last_hidden_state"][0][0])
        embedding = pt_outputs["last_hidden_state"][0][0].tolist()
        id_list.append(id)
        embedding_list.append(embedding)
        time_now = time.time()
        time_past = time_now - time_start
        print(time_past)
    data = [id_list, embedding_list]
    # 需要插入时解注释，防止误插入
    mr = collection.insert(data)
    print(mr)

if __name__ == '__main__':
    signory_insert()
    # abstract_insert()