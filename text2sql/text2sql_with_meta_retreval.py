from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain

import json
import os
import sys
import opensearchpy

##########
# BEDROCK

import boto3
from boto3 import Session
from botocore.config import Config
from botocore.exceptions import ClientError
import json

sm_client = boto3.client("sagemaker-runtime")

def get_bedrock_client(
    secret_name='chatbot_bedrock',
    asm_region_name="cn-northwest-1",
    br_region_name='us-west-2'
):
    session = Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=asm_region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])

    brrt_client = session.client(
        'bedrock-runtime',
        region_name=br_region_name,
        config=Config(retries={
            "max_attempts": 10,
            "mode": "standard",
        }),
        aws_access_key_id=secret['BEDROCK_ACCESS_KEY'],
        aws_secret_access_key=secret['BEDROCK_SECRET_KEY']
    )
    return brrt_client


def query_bedrock(prompt):
    parameters_bedrock = {
        "max_tokens_to_sample": 2048,
        "temperature": 0
    }

    brrt_client = get_bedrock_client()
    response = brrt_client.invoke_model(
      body=json.dumps({
        'prompt': f'\n\nHuman: {prompt}\n\nAssistant:',
        **parameters_bedrock
      }).encode('utf-8'),
      modelId='anthropic.claude-v2'
    )

    result = json.loads(
      response['body'].read().decode('utf-8')
    )

    return result

#########
# AOS

from boto3 import Session
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


def get_aos_client():
    aos_endpoint = 'vpc-domain66ac69e0-ebwysxh7gspk-kaihnw5p63wilhqh2ovdpj2ct4.cn-northwest-1.es.amazonaws.com.cn'
    credentials = Session().get_credentials()
    region = Session().region_name
    awsauth = AWS4Auth(credentials.access_key,
                       credentials.secret_key, region, 'es',
                       session_token=credentials.token)
    aos_client = OpenSearch(
        hosts=[{'host': aos_endpoint, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    return aos_client


def query_aos_text(query_text, index_name='prompt-optimal-index'):
    client = get_aos_client()
    query = {
        "size": 1,
        "query": {
            "query_string": {
                "default_field": "exactly_query_text",
                "query": query_text
            }
        },
        "sort": [{
           "_score": {
               "order": "desc"
           }
        }]
    }
    query_response = client.search(
        body=query,
        index=index_name
    )

    return query_response

def query_aos_text_db_table(query_text):
    response = query_aos_text(query_text)
    hits = response['hits']['hits']
    hit = hits[0]['_source']
    database_name = hit['database_name']
    table_name = hit['table_name']
    return {'database_name': database_name, 'table_name': table_name}


########
# MYSQL

from sqlalchemy import create_engine, Table, MetaData, text
from sqlalchemy.sql import text
import re

connect_string = 'mysql+pymysql://chatbot:D8JmFRDRFsAowLXyX9dXerBc@172.29.2.206:9030/'


def get_db_conn(db):
    eng = create_engine(connect_string + db)
    return eng.connect()


def get_table_info(db, table):
    conn = get_db_conn(db)
    metadata_obj = MetaData()
    tbl = Table(
        table,
        metadata_obj,
        autoload_with=conn
    )
    # 检查 table 防止注入
    table_match = re.match(r'^[a-zA-Z0-9_\.]+$', table)
    table = table_match.group(0)
    stmt = text(f'SHOW CREATE TABLE {table}')
    rs = conn.execute(stmt)

    for r in rs:
        r = r._asdict()
        show_tbl = r['Create Table']
    show_tbl_match = re.match(r'CREATE TABLE +`(?P<table_name>.+?)` +\((?P<cols>.+)\)\sENGINE.+\)', show_tbl, re.DOTALL)
    table_name = show_tbl_match['table_name']
    cols = show_tbl_match['cols']
    cols_match = re.finditer(r'`(?P<col>\S+?)`\s+(?P<type>[^\(\s]+)\S+?\s+(?:NULL|NOT NULL)? ?(?:COMMENT "(?P<comment>.+?)")?,?', cols, re.DOTALL)

    show_tbl = f'表 = {table_name}'+'\n{'
    for m in cols_match:
        m = m.groupdict()
        show_tbl += f'{m["col"]} {m["type"]} = {m["comment"]},\n'
    show_tbl += '}'

    return show_tbl

#######
# HANDLER

event = {
    'body': '{}'
}

class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__("[{}] {}".format("400",message))


def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as e:
            logger.exception(e)
            return {'Code' : 400, 'Message' : e ,'Data': "" }
        except Exception as e:
            logger.exception(e)
            return {'Code' :  500 , 'Message':"Unknown exception, please check Lambda log for more details", 'Data': ""         }

    return wrapper


def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters,
                "is_query" : True,
                "instruction" :  "为这个句子生成表示以用于检索相关文章："
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings

def aos_knn_search_v2(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'context':item['_source']['context'],'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'exactly_query_text':item['_source']['exactly_query_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose

def query_aos_knn_db_table(query,aos_endpoint,embedding_endpoint):
    query_embedding = get_vector_by_sm_endpoint(query, sm_client, embedding_endpoint_name)
    aos_client=None
    opensearch_query_response = aos_knn_search_v2(aos_client, "exactly_query_embedding",query_embedding[0], index_name, size=10)
    try:
        response = opensearch_query_response[0]
    except Exception as e:
        print(e)
        response = None
    return response


prompt_template = """
你是一个 MySQL BI 专家。
根据提供给你的表和字段，写一个 SQL 回答业务问题。
回答格式：

SQL <<<
(你写的SQL)
<<<

表和字段：
{TABLE_INFO}

问题：
{QUESTION}
"""
@handle_error
def lambda_handler(event, context):
    # json = {
    #     'question': '最近一个月温度合格的派车单数量'
    # }
    body = json.loads(event['body'])
    question = body['question']

    print('从 AOS 取回数据库和表')
    #db_tbl = query_aos_text_db_table(question)
    embedding_endpoint = os.environ.get('embedding_endpoint')
    aos_endpoint = os.environ.get('aos_endpoint')
    db_tbl = query_aos_knn_db_table(question,aos_endpoint,embedding_endpoint)

    database_name = db_tbl['database_name']
    table_name = db_tbl['table_name']

    print('使用 SQLAlchemy 取回表的字段')
    table_info = get_table_info(database_name, table_name)

    print('组装提示语')
    complete_prompt = prompt_template.format(
        TABLE_INFO=table_info,
        QUESTION=question
    )
    print(complete_prompt)

    print('使用 Bedrock-Claude 生成 SQL')
    result = query_bedrock(complete_prompt)

    print('结果 >>>>')
    print('---------------------------')
    print(result)

    return {
        'Code': 200,
        'Message': "Success",
        'Data': json.dumps(result, ensure_ascii=False)
    }

