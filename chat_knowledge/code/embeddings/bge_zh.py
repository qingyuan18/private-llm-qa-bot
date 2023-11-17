import os
import json
from typing import Any, Dict, List, Optional
import logging

from langchain.pydantic_v1 import BaseModel
from langchain.schema.embeddings import Embeddings

import boto3


logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker-runtime')
endpoint_name = os.environ.get('embedding_endpoint')


class BgeZhEmbeddings(BaseModel, Embeddings):
    """BgeZh embedding models. OpenAI Embedding - compatible

    To use, you should have the ``openai`` python package installed.
    """


    def _embedding_func(self, text: str) -> List[float]:
        """Call out to Bedrock embedding endpoint."""

        parameters = {

        }

        instruction = "为这个句子生成表示以用于检索相关文章："

        response_model = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(
                {
                    "inputs": text,
                    "parameters": parameters,
                    "is_query" : True,
                    "instruction" :  instruction,
                }
            ),
            ContentType="application/json",
        )
        json_str = response_model['Body'].read().decode('utf8')
        json_obj = json.loads(json_str)
        embeddings = json_obj['sentence_embeddings'][0]

        return embeddings


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a Bge-zh model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        for text in texts:
            response = self._embedding_func(text)
            results.append(response)
        return results

    
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Bge-zh model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embedding_func(text)