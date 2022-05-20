# -*- coding:utf-8 -*-
from transformers import RoFormerModel, RoFormerTokenizer
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import pymysql
from pymilvus import Collection
import time
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dunum =[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_query():
    connections.connect(host='localhost', port='19530')
    collection = Collection("signory")      # Get an existing collection.
    collection.load()
    query = "一种推进剂，包括燃料，氧化剂，其特征在于采用五氧化二氮做氧化剂，金属氢化物为燃料，两者反应，金属氢化物加热分解，释放大量的氢气，具有很高比冲量的效果，提供足够能量。"

    tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_sim_char_ft_base")
    # print(tokenizer.tokenize(query))
    pt_inputs = tokenizer(query, max_length=64, padding='max_length', return_tensors="pt")
    pt_outputs = pt_model(**pt_inputs)
    query_embeddings = [pt_outputs["last_hidden_state"][0][0].tolist()]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    vec_results = collection.search(query_embeddings, "signory", param=search_params, limit=9, expr=None)
    similar_titles = []
    print("query: " + query)

    if vec_results[0][0].distance < 1000:
        print("There are no similar questions in the database, here are the closest matches:")
    else:
        print("There are similar questions in the database, here are the closest matches: ")

    # 通过id，从mysql中查对应的title等信息
    mysqlconnection = pymysql.connect(host='10.1.0.177',
                                     user='bwj',
                                     password='bwj',
                                     database='patent',
                                     cursorclass=pymysql.cursors.DictCursor)

    results = []
    with mysqlconnection:
        with mysqlconnection.cursor() as cursor:
            # Read a single record
            for result in vec_results[0]:
                sql = "SELECT  `patent_id`, `signory` FROM `signory`" + " where signory_id = " + str(result.signory_id) + ";"
                cursor.execute(sql)
                rows = cursor.fetchall()
                if len(rows):
                    results.append({'id': rows[0]['id'], 'distance': result.distance, 'title': rows[0]['title']})
    for result in results:
        print(result)

def test_speed():
    tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_sim_char_ft_base")
    pt_model = torch.nn.DataParallel(pt_model, device_ids=dunum)
    pt_model.to(device)
    query = "一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。一种乳膨炸药，其特征在于，它由氧化剂，水，可燃剂，乳化剂制成，各原料组分的重量百分配比为：氧化剂84-92，水3-12，可燃剂4-10，乳化剂0.5-1.5。"
    print(len(query))
    time_start = time.time()
    pt_inputs = tokenizer(query, max_length=64, padding='max_length', return_tensors="pt")
    pt_outputs = pt_model(**pt_inputs)
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)

def test_encode():
    tokenizer = RoFormerTokenizer.from_pretrained("./roformer_chinese_sim_char_ft_base")
    pt_model = RoFormerModel.from_pretrained("./roformer_chinese_sim_char_ft_base")
    query = "举头望明月"
    # query = "为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。其中，最典型的就是数据的归一化处理"
    pt_inputs = tokenizer(query, max_length=505, return_tensors="pt")
    pt_outputs = pt_model(**pt_inputs)
    # print(pt_outputs["last_hidden_state"][0][0])
    embedding = pt_outputs["last_hidden_state"][0][0].tolist()
    print(query)
    print(embedding)

if __name__ == '__main__':
    test_encode()
    # test_query()
    # test_speed()
    # print(torch.version.cuda)