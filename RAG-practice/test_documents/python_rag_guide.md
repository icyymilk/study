\# Python RAG开发指南（2024版）

\## 一、核心步骤

1\. 文档加载：使用LangChain的PyPDFLoader/PythonLoader加载本地文档，支持PDF/TXT/MD等格式；

2\. 文本预处理：

&nbsp;  - 中文文本：按「。！？」分割，chunk\_size=1000，chunk\_overlap=100；

&nbsp;  - 英文文本：按句子分割，chunk\_size=1500，chunk\_overlap=150；

3\. 嵌入模型：

&nbsp;  - 中文推荐：BAAI/bge-small-zh-v1.5、moka-ai/m3e-small；

&nbsp;  - 英文推荐：all-MiniLM-L6-v2、BAAI/bge-small-en-v1.5；

4\. 向量库构建：优先使用Chroma（轻量）、Qdrant（生产级），需设置persist\_directory持久化；

5\. 检索器配置：

&nbsp;  - 基础检索：similarity\_search，k=3~5；

&nbsp;  - 优化检索：MMR（search\_type="mmr"），提升结果多样性；

6\. 生成环节：

&nbsp;  - 大模型推荐：Qwen-7B-Chat、ChatGLM4、Llama-3-8B-Chinese；

&nbsp;  - Prompt模板需适配中文表达，避免翻译腔。



\## 二、常见问题解决

\### 问题1：检索结果不相关

解决方法：

\- 更换高精度嵌入模型（如bge-base-zh-v1.5）；

\- 启用CrossEncoder重排序（BAAI/bge-reranker-base）；

\- 调整文本分割粒度（减小chunk\_size至800）。



\### 问题2：生成速度慢

解决方法：

\- 使用vllm加速推理，tensor\_parallel\_size=1（单卡）；

\- 启用4bit量化（bitsandbytes），降低显存占用；

\- 减小max\_tokens至500以内。



\## 三、核心代码示例

```python

\# 初始化中文嵌入模型

from langchain\_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(

&nbsp;   model\_name="BAAI/bge-small-zh-v1.5",

&nbsp;   model\_kwargs={"device": "cuda"},

&nbsp;   encode\_kwargs={"normalize\_embeddings": True}

)

