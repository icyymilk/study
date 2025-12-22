print("=== RAG 基础功能测试 ===")

# 修正路径：test_documents 不是 test_docoments
from langchain_community.document_loaders import TextLoader  # 使用新导入方式
loader = TextLoader('test_documents/eu_ai_act_simplified.txt', encoding='utf-8')
docs = loader.load()
print(f"✅ 文档加载成功: {len(docs)} 个文档")

# 测试文本分割
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = splitter.split_documents(docs)
print(f"✅ 文本分割成功: {len(texts)} 个片段")

# 测试向量化（使用正确的导入方式）
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    test_embedding = embeddings.embed_query("测试文本")
    print(f"✅ 向量化成功: 维度 {len(test_embedding)}")
except ImportError:
    # 备选方案
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    test_embedding = model.encode(["测试文本"])
    print(f"✅ 向量化成功 (直接使用sentence-transformers): 维度 {test_embedding.shape[1]}")

print("\n🎉 基础RAG流程测试通过！可以开始实施优化了。")