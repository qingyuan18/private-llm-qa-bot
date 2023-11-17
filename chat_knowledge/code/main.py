import os
import json
import logging

from retrievers.opensearch import AmazonOpenSearchRetriever
from prompts.qa_claude2 import QaClaude2Prompts
from llms.claude import Claude2LLM


logger = logging.getLogger()
logger.setLevel(logging.INFO)

        
def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.exception(err)
            return json.dumps({
                'Code': 500,
                'Message': 'Unknown exception, please check Lambda log for more details',
                'Data': ''
            })

    return wrapper


def parameter_validate(model_name):
    """
    model_name: must be 'claude2'
    """
    if model_name != 'claude2':
        return False

    return True


@handle_error
def lambda_handler(event, context):
    """
    event:
    context:
    """
    logger.info(f'Lambda event : {event}, context : {context}')

    # 1. Initialize the parameters
    # 1.1 Retrieve from 'body'
    request_body = event.get('body')
    if request_body:
        request_body = json.loads(request_body)

    query = request_body['prompt']
    model_name = request_body['model']
    max_tokens = int(request_body.get('max_tokens', 2048))
    temperature = float(request_body.get('temperature', 0.1))
    top_p = float(request_body.get('top_p', 0.95))

    # 1.2 Retrieve from environment
    aos_endpoint = os.environ.get('aos_endpoint', '')
    aos_index = os.environ.get('aos_index', '')

    bedrock_runtime_region = os.environ.get('bedrock_runtime_region', '')
    bedrock_aksk_region = os.environ.get('bedrock_aksk_region', '')
    bedrock_secret_name = os.environ.get('bedrock_secret_name', '')

    logger.info('#' * 50)
    logger.info(f'query                  :{query}')
    logger.info(f'model_name             :{model_name}')
    logger.info(f'max_tokens             :{max_tokens}')
    logger.info(f'temperature            :{temperature}')
    logger.info(f'top_p                  :{top_p}')
    logger.info(f'aos_endpoint           :{aos_endpoint}')
    logger.info(f'aos_index              :{aos_index}')
    logger.info(f'bedrock_runtime_region :{bedrock_runtime_region}')
    logger.info(f'bedrock_aksk_region    :{bedrock_aksk_region}')
    logger.info(f'bedrock_secret_name    :{bedrock_secret_name}')
    logger.info('#' * 50)

    # 1.3 Parameter validation
    if not parameter_validate(model_name):
        raise Exception('Parameter is invalid.')

    # 2. Retriever(Vector DB/OpenSearch) to search similar results.
    # 2.1 Initialize 'retriever/OpenSearch'
    retriever = AmazonOpenSearchRetriever()
    
    # 2.2 Create index if necessary
    retriever.create_document_index()

    # 2.3 Recall the results
    context = retriever.get_relevant_documents(query)
    # context = ''
    logger.info(f'context:{context}')

    # 3. Generate appropriate prompts
    template = QaClaude2Prompts()
    final_prompts = template.generate_prompt(query, context)
    logger.info(f'final_prompts:{final_prompts}')

    # 4. LLM to summary/generate answers based on sources.
    # 4.1 Initialize LLM
    llm = Claude2LLM(max_tokens=max_tokens,
                     temperature=temperature,
                     top_p=top_p,
                     stop_sequences='')
    llm.initial_llm(region=region,
                    prompt_template=template.generate_template())

    # 4.2 Generate answers
    answer = llm.generate(query, context)
    logger.info(f'answer:{answer}')

    # 5. Return the results
    return json.dumps({
        'Code': 200,
        'Message': 'success',
        'Data': answer
    })
