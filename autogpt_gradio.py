import os
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
import gradio as gr

def agent_run(input_question):
    
    output_response = agent.run(input_question)
    return output_response

def launch_gradio():

    iface = gr.Interface(
        fn=agent_run,
        title="大作业3 autogpt图形化",
        inputs=[
            gr.Textbox(label="请输入你的问题")
        ],
        outputs=[
             gr.Textbox(label="AutoGPT的回答")
        ],
        allow_flagging="never"
    )

    #iface.launch(share=True, server_name="0.0.0.0")
    iface.launch()
    
def initialize_autogpt():
    os.environ["SERPAPI_API_KEY"] = "e65622355785aba531fe0f3733c6c429e3ec43457c916a0c3006e6f81d433369"
    # OpenAI Embedding 模型
    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    global agent
    agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
)

if __name__ == "__main__":
    # 初始化 translator
    initialize_autogpt()
    # 启动 Gradio 服务
    launch_gradio()