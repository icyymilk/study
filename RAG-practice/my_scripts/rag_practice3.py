from chromadb.config import Settings
import chromadb
from colorama import Fore,Style
from pathlib import Path
from chromadb.utils import embedding_functions
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
#Create ChromaDB and load data
#为什么选择ChromaDB:Python原生，适用于开发和生产的测试环境，API是pythonic，适合入门并且可以达到生产规模。
def buildDB():
    client = chromadb.PersistentClient(path="./chroma_db")
    #模型外部实例化
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #库是client，分库是collection
    try:
        client.delete_collection("job_ads")
    except:
        pass
    #创建一个分库
    collection = client.create_collection(
        name = "job_ads",
        metadata = {"description":"Job advertisement corpus"}
    )
    print("Collection Build Successfully!")
    #方法一：先在外部对文档库向量化再入库
    #先读入二进制数据
    corpus=Path("example_corpus")
    job_ads = {}
    for file_path in sorted(corpus.glob("job_ad_*.txt")):
        with open(file_path,'r',encoding='utf-8') as f:
            job_ads[file_path.stem] = f.read()

    ids = list(job_ads.keys())
    documents = list(job_ads.values())
    embeddings = model.encode(documents,show_progress_bar = True)

    metadatas = []
    for i,(name,text) in enumerate(zip(ids,documents)):
        word_count = len(text.split())
        char_count = len(text)

        metadatas.append({
            'doc_id':name,
            "word_count" :word_count,
            "char_count":char_count,
            "index":i
        })
    collection.add(
        ids = ids,
        documents = documents,
        embeddings = embeddings.tolist(),
        metadatas = metadatas
    )
    #完成，此时获得了一个分库，里面载入了六个文本数据，验证一下
    print(f"Connected to collection with collection{collection.count()}documents")
    
    #方法二：直接运用Chromamdb对应的内部onix源嵌入模型,需要在创立分库时指定库嵌入函数
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'
    )
    client = chromadb.PersistentClient(path="./chroma.db")
    collection2 = client.create_collection(
        name = "job_ads2",
        metadata={"description":"Inner Ef added database"},
        embedding_function = ef,
    )
    #分库格式初始化用add
    #分库后续插入用upsert
    #删除用delete,用get获取对应ids项
    collection2.add(
        ids = ids,
        documents = documents,
        metadatas = metadatas
    )
    collections = client.list_collections()
    table = []
    for col in collections:
        table.append([col.name,col.count(),col.metadata])
    print("Collection Checkout")
    print(tabulate(table,headers=["Name","Counts","Metadata"],tablefmt = 'grid'))

    collection.modify(
        metadata={
            "description":"Job advertisements corpus",
            "last_updated":time.time()
        }
    )

    query = "software engineer position"

    query_embeddings = model.encode(query) #->[384]
    start = time.time()
    #Expected embeddings to be a list of floats or ints, a list of lists, a numpy array, or a list of numpy arrays,
    results = collection.query(
        query_embeddings = [query_embeddings.tolist()], #[1,384] 非张量
        n_results = 5
    )
    query_time = time.time()-start
    print(f"单Query查询:{query}")
    print(f"Query Time:{query_time*1000:.2f}ms")
    print(f"Document searched:{collection.count()}")
    print(f"Results returned:{len(results['ids'][0])}")
    for i,(doc_id,distance) in  enumerate(zip(results["ids"][0],results["distances"][0]),1):
        metadata = results['metadatas'][0][i-1]
        preview = results["documents"][0][i-1][:100].replace('\n',' ')
        print(f"  {i}. {doc_id}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Words: {metadata['word_count']}")
        print(f"     Preview: {preview}...")
    #model.encode 的行为：单字符串→一维数组，字符串列表→二维数组，这是 Sentence-BERT 等模型的通用规则。
    queries  = [
        "Python developer",
        "web development position"
    ]
    
    embeddings2 = model.encode(queries) # [3,384]
    print(embeddings2.shape)
    start = time.time()
    #为什么要tolist？
    #一维数组（通常指 numpy.ndarray）和列表（list）虽然看起来都是 “一串元素”，但底层设计、功能、性能和使用场景有本质区别。
    #类型	本质	维度表示（384 个元素）	示例
    # Numpy 一维数组	同质、连续的内存块	(384,)（注意逗号）	np.array([1.2, 3.4, ..., 9.8])
    # Python 原生列表	异构、分散的对象引用	[384]（无逗号）	[1.2, 3.4, ..., 9.8]
    # 向量数据库（如 Milvus/Weaviate/Qdrant）的 query_embeddings 参数：✅ 支持列表 / 普通 Python 数组（因为是通用数据格式，易序列化 / 传输）；❌ 不直接支持 numpy.ndarray（避免依赖 numpy，且跨语言调用时兼容性差）。
    # 简单说：numpy数组 是 “计算友好型”，列表 是 “接口友好型”，转列表是为了适配向量数据库的输入要求
    results2 = collection.query(
        query_embeddings= embeddings2.tolist(),
        n_results=5
    )
    
    queries_time = time.time()-start
    print(f"多Query查询:{queries}")
    print(f"Query Time:{queries_time*1000:.2f}ms")
    print(f"Document searched:{collection.count()}")
    print(f"Results returned:{len(results2['ids'][0])}")
    for i,(doc_id,distance) in  enumerate(zip(results2["ids"][0],results2["distances"][0]),1):
        metadata = results2['metadatas'][0][i-1]
        preview = results2["documents"][0][i-1][:100].replace('\n',' ')
        print(f"  {i}. {doc_id}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Words: {metadata['word_count']}")
        print(f"     Preview: {preview}...")
    for i,(doc_id,distance) in  enumerate(zip(results2["ids"][1],results2["distances"][1]),1):
        metadata = results2['metadatas'][1][i-1]
        preview = results2["documents"][1][i-1][:100].replace('\n',' ')
        print(f"  {i}. {doc_id}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Words: {metadata['word_count']}")
        print(f"     Preview: {preview}...")

if __name__ == "__main__":
    buildDB()