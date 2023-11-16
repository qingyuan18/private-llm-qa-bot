from langchain.prompts import PromptTemplate


class QaClaude2Prompts():
    """
    Prompt template for Claude2 Q&A
    """
    template = """Human\n\n以下context内的文本内容为背景知识:\n<context>\n{context}\n</context>\n请根据背景知识,回答这个问题,如果context内的文本内容为空,则请专业人士进行解答.\n{question}\n\nAssistant:"""

    prompt_template = PromptTemplate(
        template=template,
        partial_variables={},
        input_variables=["context", "question"]
    )


    def generate_template(self):
        """
        Generate templates
        """
        return self.prompt_template


    def generate_prompt(self, prompt, context):
        """
        context:
        prompt:
        """  
        return self.prompt_template.format(context=context, question=prompt)