import os
import json
import logging

from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Claude2LLM():
    """
    Claude2 LLM
    """

    def __init__(self,
                 max_tokens,
                 temperature,
                 top_p,
                 stop_sequences) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.llm = None
        self.llmchain = None
    

    def _get_bedrock_aksk(self,
                          secret_name=os.environ.get('bedrock_secret_name', 'chatbot_bedrock'),
                          region_name=os.environ.get('bedrock_aksk_region', 'us-west-2')):
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)                           
        except ClientError as err:
            raise err
        
        # Decrypts secret using the associated KMS key.
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']
    

    def initial_llm(self, region, prompt_template):
        """
        region: region name, default is 'us-west-2'
        prompt_template
        """
        # Retrieve 'Bedrock' AKSK.
        aws_access_key_id, aws_secret_access_key = self._get_bedrock_aksk()

        # Define bedrock client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('bedrock_runtime_region', region),
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Define parameters
        parameters = {
            'max_tokens_to_sample': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            # 'stop_sequences': self.stop_sequences
        }

        # Define LLM - Bedrock
        self.llm = Bedrock(model_id='anthropic.claude-v2',
                      client=bedrock_client,
                      streaming=False,
                      model_kwargs=parameters)
        
        # Define LLMChain
        self.llmchain = LLMChain(llm=self.llm, verbose=False, prompt=prompt_template)
        
    

    def generate(self, query, context):
        """
        query: Origin question
        context: Retrival from OpenSearch
        """
        return self.llmchain.run({'question': query, 'context': context})
