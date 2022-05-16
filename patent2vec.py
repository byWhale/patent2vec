#Connectings to Milvus, BERT and Postgresql
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import pymysql
connections.connect(host='localhost', port='19530')

TABLE_NAME = "patent"

id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
title = FieldSchema(name="title", dtype=DataType.FLOAT_VECTOR, dim=768)
schema = CollectionSchema(fields=[id,title], description="title of patent")
collection = Collection(name=TABLE_NAME, schema=schema)

index_param = {
        "metric_type":"L2",
        "index_type":"IVF_SQ8",
        "params":{"nlist":1024}
    }

collection.create_index(field_name="title", index_params=index_param)