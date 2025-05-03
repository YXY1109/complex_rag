from typing import Dict, List, Optional

from elasticsearch import Elasticsearch, helpers

from src.config.config import settings
from src.utils.logger import logger


class ElasticSearchClient:
    def __init__(self):

        try:
            # 修正端口变量名拼写错误
            es_host = settings.ES_HOST
            es_port = settings.ES_PORT
            es_user = settings.ES_USER
            es_password = settings.ES_PASSWORD

            self.es_index = settings.ES_INDEX
            self.client = Elasticsearch(
                [f"http://{es_host}:{es_port}"],  # noqa
                basic_auth=(es_user, es_password),
                request_timeout=30,  # 设置请求超时时间
            )
            # 测试连接
            if self.client.ping():
                logger.info("Ping OK，Connected to Elasticsearch")
            else:
                logger.error("Failed to connect to Elasticsearch")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")

    def close(self):
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing Elasticsearch connection: {e}")

    def list_index(self) -> List[Dict]:
        """
        列出索引（知识库）
        """
        try:
            return self.client.cat.indices(format="json")
        except Exception as e:
            logger.error(f"Error listing indices: {e}")
            return []

    def has_index(self, index_name: str) -> bool:
        try:
            return self.client.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"Error checking if index {index_name} exists: {e}")
            return False

    def create_index(self, index_name: str):
        """
        创建索引（知识库），指定索引的中文分词分析器  自动创建
        安装ik_smart:进入容器内，执行命令
        bin/elasticsearch-plugin install https://get.infini.cloud/elasticsearch/analysis-ik/8.17.4
        """
        index_name = index_name.lower()
        try:
            if self.has_index(index_name):
                logger.info(f"Index {index_name} already exists.")
                return

            # 提取设置和映射为类属性，提高复用性
            es_settings = self._get_es_settings()
            es_mappings = self._get_es_mappings()

            logger.info(f"Creating index {index_name} with:\nmappings: {es_mappings}\nsettings: {es_settings}")
            self.client.indices.create(index=index_name, mappings=es_mappings, settings=es_settings)
            logger.info(f"Index {index_name} created successfully.")
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")

    @staticmethod
    def _get_es_settings() -> Dict:
        """
        获取 Elasticsearch 索引设置
        """
        return {"index": {"similarity": {"custom_bm25": {"type": "BM25", "k1": 1.3, "b": 0.6}}}}

    @staticmethod
    def _get_es_mappings() -> Dict:
        """
        获取 Elasticsearch 索引映射
        """
        return {
            "properties": {
                "file_id": {"type": "keyword", "index": True},
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",
                    "index": True,
                    "analyzer": "ik_smart",
                    "search_analyzer": "ik_smart",
                },
            }
        }

    def delete_index(self, index_name: str) -> bool:
        """
        删除索引（知识库）
        """
        try:
            self.client.indices.delete(index=index_name, ignore_unavailable=True)
            logger.info(f"success to delete index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"fail to delete: {index_name}\nERROR: {e}")
            return False

    def insert(self, insert_list: List[Dict], index_name: str, refresh: bool = False) -> bool:
        """
        插入文档
        #async
        """
        index_name = index_name.lower()
        try:
            self.create_index(index_name)
        except Exception as e:
            logger.error(f"创建索引 {index_name} 时出错: {e}")
            return False

        actions = []
        for item in insert_list:
            if not isinstance(item, dict) or "metadata" not in item or "chunk_id" not in item["metadata"]:
                logger.warning(f"跳过无效文档: {item}。文档必须是字典且包含 metadata.chunk_id。")
                continue
            action = {"_op_type": "index", "_id": item["metadata"]["chunk_id"]}
            action.update(item)
            actions.append(action)

        if not actions:
            logger.warning("没有有效的文档可以插入。")
            return False

        try:
            documents_written, errors = helpers.bulk(
                client=self.client,
                actions=actions,
                refresh=refresh,  # 直接使用 refresh 参数
                index=index_name,
                stats_only=True,
                raise_on_error=True,
            )
            if documents_written > 0:
                logger.info(f"成功插入 {documents_written} 个文档到索引 {index_name}。")
            if errors:
                logger.error(f"插入文档到索引 {index_name} 时出现 {len(errors)} 个错误: {errors}")
            return True
        except Exception as e:
            logger.error(f"执行批量插入操作到索引 {index_name} 时出错: {e}")
            return False

    def delete_by_chunk_ids(self, index_name: Optional[str] = None, ids: Optional[List[str]] = None) -> str:
        """
        删除文档
        """
        if index_name is None or ids is None:
            return "No chunks to delete."

        index_name = index_name.lower()
        try:
            helpers.bulk(
                client=self.client,
                actions=({"_op_type": "delete", "_id": id_} for id_ in ids),
                refresh="wait_for",
                index=index_name,
                stats_only=True,
                raise_on_error=False,
                ignore_status=404,
            )
            logger.info(f"success to delete chunks ids: {ids} from index: {index_name}")
        except Exception as e:
            logger.error(f"Error delete chunks: {e}")
        return f"success to delete chunks: {ids} in index: {index_name}"

    def build_query_body(self, query: str, field: str, top_k: int) -> Dict:
        """
        构建 Elasticsearch 查询体
        """
        if field == "content":
            return {"query": {"match": {"content": {"query": query, "fuzziness": "AUTO"}}}, "size": top_k}
        elif field == "file_id":
            return {"query": {"term": {"file_id": query}}}
        else:
            raise ValueError(f">>es>> - Please provide valid field: {field}")

    def search(self, queries: List[str], index_name: str, top_k: int = 10, field: str = "content") -> List[Dict]:
        """
        检索文档
        """
        fields = ["file_id", "content", "metadata"]
        search_results = []
        search_item_seen = set()
        for query in queries:
            try:
                query_body = self.build_query_body(query, field, top_k)
                response = self.client.search(index=index_name, source={"includes": fields}, **query_body)
                for hit in response["hits"]["hits"]:
                    search_tag = f"{hit['_index']}_{hit['_id']}"
                    if search_tag in search_item_seen:
                        continue
                    search_item_seen.add(search_tag)

                    search_item = {"index": hit["_index"], "id": hit["_id"], "score": hit["_score"]}

                    for f in fields:
                        search_item[f] = hit["_source"].get(f)

                    search_results.append(search_item)
            except Exception as e:
                logger.error(f"Error searching for query {query} in index {index_name}: {e}")

        # search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results

    def fetch_all(self, index_name: str) -> List[Dict]:
        query = {"query": {"match_all": {}}}
        all_documents = []
        try:
            response = self.client.search(index=index_name, body=query, scroll="2m", size=1000, sort="_doc")
            scroll_id = response["_scroll_id"]
            all_documents.extend(response["hits"]["hits"])

            while len(response["hits"]["hits"]) > 0:
                response = self.client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                all_documents.extend(response["hits"]["hits"])

            logger.info(f"Retrieved es all {len(all_documents)} documents.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

        return all_documents


if __name__ == "__main__":
    client = ElasticSearchClient()
    index_list = client.list_index()
    name = "test"
    print(index_list)
    print(client.has_index(name))

    # client.delete_index(name)
    # client.create_index(name)
    # client.insert([
    #     {'metadata': {'chunk_id': '1'}, 'content': '今年是“十四五”规划收官之年、“十五五”规划谋篇布局之年'},
    #     {'metadata': {'chunk_id': '2'},
    #      'content': '4月30日，习近平总书记在上海主持召开部分省区市“十五五”时期经济社会发展座谈会，为进一步全面深化改革、推动中国式现代化行稳致远筹谋。'},
    # ], name)
    # print("插入完成")

    data = client.search(["十五五"], name)
    print(data)
