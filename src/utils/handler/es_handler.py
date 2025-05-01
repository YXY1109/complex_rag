from elasticsearch import Elasticsearch

from src.config.config import settings

if __name__ == "__main__":
    es_host = settings.ES_HOST
    es_prot = settings.ES_PORT
    es_user = settings.ES_USER
    es_password = settings.ES_PASSWORD
    es_index = settings.ES_INDEX

    client = Elasticsearch([f"http://{es_host}:{es_prot}"], basic_auth=(es_user, es_password))

    # 测试连接
    if client.ping():
        print("Connected to Elasticsearch")
    else:
        print("Failed to connect")

    # 示例操作：获取集群健康状态
    info = client.cluster.health()
    print(info)

    res1 = client.index(index="my-index", id="1", body={"name": "John Doe", "age": 30})
    print(res1)
    print(res1["result"])
    print("*" * 20)
    res2 = client.search(index="my-index", query={"match": {"name": "John"}})
    print(res2)
    print(res2["hits"]["hits"])
    print(res2.body.get("hits").get("hits"))
