import os
from typing import List, Optional, Dict, Any
import logging

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import root_validator, validator
from langchain.schema import BaseRetriever, Document

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import AuthorizationException

from embeddings.bge_zh import BgeZhEmbeddings


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AmazonOpenSearchRetriever(BaseRetriever):
    """`Amazon OpenSearch` retriever.

    Args:
        index_id: OpenSearch index id

        aos_endpoint: OpenSearch Endpoint

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        top_k: No of results to return

        client: client for OpenSearch

        user_context: Provides information about the user context
            See: https://docs.aws.amazon.com/kendra/latest/APIReference

    Example:
        .. code-block:: python

            retriever = AmazonOpenSearchRetriever(
                index_id="c0806df7-e76b-4bce-9b5c-d5582f6b1a03"
            )

    """

    index_id: str = os.environ.get('aos_index', '')
    aos_endpoint: str = os.environ.get('aos_endpoint', '')
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    top_k: int = 3
    aos_knn_threshold: float = 0.4
    aos_match_threshold: float = 1.0
    client: Any
    user_context: Optional[Dict] = None
    embeddings_model = BgeZhEmbeddings()

    
    def _create_client(self):

        try:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()

            client_params = {}
            client_params["region_name"] = session.region_name

            aws_auth = AWS4Auth(credentials.access_key,
                                credentials.secret_key,
                                client_params["region_name"],
                                'es',
                                session_token=credentials.token)

            self.client = OpenSearch(
                hosts=[{"host": self.aos_endpoint, "port": 443}],
                http_auth=aws_auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )

            logger.info(f'self.index_id    :{self.index_id}')
            logger.info(f'self.aos_endpoint:{self.aos_endpoint}')
            logger.info(f'self.client      :{self.client}')

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e


    def create_document_index(self):
        from opensearchpy.exceptions import RequestError

        payload = {
            "settings" : {
                "index":{
                    "number_of_shards" : 1,
                    "number_of_replicas" : 0,
                    "knn": "true",
                    "knn.algo_param.ef_search": 32
                }
            },
            "mappings": {
                "properties": {
                    "publish_date" : {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss"
                    },
                    "idx" : {
                        "type": "integer"
                    },
                    "doc_type" : {
                        "type" : "keyword"
                    },
                    "doc": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "doc_title": {
                        "type": "keyword"
                    },
                    "doc_category": {
                        "type": "keyword"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    }
                }
            }
        }
    
        self._create_client()
        index_exist = self.client.indices.exists(index=self.index_id)
        if not index_exist:
            try:
                self.client.indices.create(index=self.index_id, body=payload)
            except RequestError as err:
                logger.error(err)
                raise ValueError(
                    "Index already exists."
                ) from err

    def _search_with_aos_knn(self, embed_vector, size=6):
        
        try:
            aos_knn_responses = []

            query = {
                "size": size,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embed_vector,
                            "k": size
                        }
                    }
                }
            }

            aos_knn_response = self.client.search(
                body=query,
                index=self.index_id
            )

            aos_knn_responses = [
                {'idx': item['_source'].get('idx', 1),
                'doc_category': item['_source']['doc_category'],
                'doc_title': item['_source']['doc_title'],
                'id': item['_id'],
                'doc': "{}=>{}".format(item['_source']['doc'], item['_source']['content']),
                "doc_type": item["_source"]["doc_type"],
                "score": item["_score"]} 
                for item in aos_knn_response["hits"]["hits"]
            ]

            return aos_knn_responses
        
        except AuthorizationException as err:
            logger.error(f"AuthorizationException : {err.error}")
            return aos_knn_responses


    def _search_with_aos_match(self, query_term, exactly_match=False, size=6):
         
        try:
            aos_match_responses = []
            query = None
            
            if exactly_match:
                query = {
                    "size": size,
                    "query": {
                        "match_phrase": {
                            "doc": {
                                "query": query_term,
                                "analyzer": "ik_smart",
                                "slop": 3,
                            },
                        },
                    }
                }
            else:
                query = {
                    "size": size,
                    "sort": [{
                        "_score": {
                            "order": "desc"
                        }
                    }],
                    "query": {
                        "bool": {
                            "minimum_should_match": 1,
                            "should": [{
                                    "bool": {
                                        "must": [{
                                                "term": {
                                                    "doc_type": "Question"
                                                }
                                            },
                                            {
                                                "match": {
                                                    "doc": query_term
                                                }
                                            }
                                        ]
                                    }
                                },
                                {
                                    "bool": {
                                        "must": [{
                                                "term": {
                                                    "doc_type": "Paragraph"
                                                }
                                            },
                                            {
                                                "match": {
                                                    "content": query_term
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                }

            query_response = self.client.search(
                body=query,
                index=self.index_id
            )

            if exactly_match:
                aos_match_responses = [ 
                    {'idx': item['_source'].get('idx',0), 
                    'doc_category': item['_source']['doc_category'], 
                    'doc_title':item['_source']['doc_title'], 
                    'id': item['_id'], 
                    'doc': item['_source']['content'], 
                    'doc_type': item['_source']['doc_type'], 
                    'score': item['_score']} 
                    for item in query_response["hits"]["hits"]]
            else:
                aos_match_responses = [ 
                    {'idx': item['_source'].get('idx',0), 
                    'doc_category': item['_source']['doc_category'], 
                    'doc_title': item['_source']['doc_title'], 
                    'id':item['_id'], 
                    'doc':"{}=>{}".format(item['_source']['doc'], item['_source']['content']), 
                    'doc_type': item['_source']['doc_type'], 
                    'score': item['_score']} 
                    for item in query_response["hits"]["hits"]]

            return aos_match_responses

        except AuthorizationException as err:
            logger.error(f"AuthorizationException : {err.error}")
        
        return aos_match_responses
    

    def _combine_recalls(self, aos_knn_responses, aos_match_responses):
    
        # 1. Filter with threshold
        aos_knn_responses_filter = [ item for item in aos_knn_responses if item["score"] > self.aos_knn_threshold ]
        aos_match_responses_filter = [ item for item in aos_match_responses if item["score"] > self.aos_match_threshold ]
        
        # 2. Drop duplicated items
        unique_ids = set()
        aos_responses = []
        for item in aos_knn_responses_filter:
            if item["id"] not in aos_responses:
                aos_responses.append(item)
                unique_ids.add(item["id"])
        
        return aos_responses, aos_knn_responses_filter, aos_match_responses_filter
    

    def _qa_knowledge_fewershot_build(self, recalls):
        ret_context = []
        
        for recall in recalls:
            if recall['doc_type'] == 'Question':
                q, a = recall['doc'].split("=>")
                qa_example = "{}: {}\n{}: {}".format("问题", q, "回答", a)
                ret_context.append(qa_example)

        context_str = "\n\n".join(ret_context)

        return context_str


    def _get_relevant_documents(
            self, 
            query: str, 
            *, 
            run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Run search on OpenSearch index and get top k documents

        Example:
        .. code-block:: python

            docs = retriever.get_relevant_documents('This is my query')

        """
        # 1. Invoke embedding model to embed the query.
        if isinstance(query, str):
            response = self.embeddings_model.embed_query(query)
        else:
            response = self.embeddings_model.embed_documents(query)

        # 2. Invoke retriever(Vector DB/OpenSearch) to search similar results.
        # 2.1 'K-NN' search
        aos_knn_responses = self._search_with_aos_knn(embed_vector=response, size=2)

        # 2.2 'Match' search
        aos_match_responses = self._search_with_aos_match(query_term=query, size=2)

        # 2.3 Merge 'K-NN' and 'Match' search results
        aos_responses, _, _ = self._combine_recalls(aos_knn_responses, aos_match_responses)

        # 2.4 Generate fewer shot context
        context = self._qa_knowledge_fewershot_build(aos_responses)

        top_k_results = []
        top_k_results.append(Document(page_content=context))
        
        return top_k_results
