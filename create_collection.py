#Connectings to Milvus, BERT and Postgresql
from pymilvus import connections
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import pymysql

def create_title_collection():
    connections.connect(host='localhost', port='19530')
    TABLE_NAME = "title"
    id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    title = FieldSchema(name="title", dtype=DataType.FLOAT_VECTOR, dim=768)
    schema = CollectionSchema(fields=[id,title], description="title of patent")
    collection = Collection(name=TABLE_NAME, schema=schema)

    index_param = {
            "metric_type":"L2",
            "index_type":"IVF_SQ8",
            "params":{"nlist":1024}
        }
    collection.create_index(field_name="title", index_params=index_param)


def create_signory_collection():
    connections.connect(host='localhost', port='19530')
    TABLE_NAME = "signory"
    signory_id = FieldSchema(name="signory_id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    patent_id = FieldSchema(name="patent_id", dtype=DataType.INT64, is_primary=False, auto_id=False)
    signory = FieldSchema(name="signory", dtype=DataType.FLOAT_VECTOR, dim=768)
    schema = CollectionSchema(fields=[signory_id, patent_id, signory], description="signory_seg of patent")
    collection = Collection(name=TABLE_NAME, schema=schema)

    index_param = {
            "metric_type":"L2",
            "index_type":"IVF_SQ8",
            "params":{"nlist":1024}
        }
    collection.create_index(field_name="signory", index_params=index_param)

if __name__ == '__main__':
    create_signory_collection()