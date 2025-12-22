import numpy as np
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style
import os 
import time
from pathlib import Path

def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}\n")

def operation():
    
    
    examples = [
        "I love machine learning",
        "I enjoy artificial intelligence",
        "The weather is nice today"
    ]
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. '{ex}'")

    sentences = [
        "I love cats",
        "I love dogs",
        "Cats and dogs"
    ]
    #去除重复的，并按字母排序
    vocab = sorted(set(' '.join(sentences).lower().split()))
    print(vocab)
    for sent in sentences:
        words = sent.lower().split()
        vector = [words.count(word) for word in vocab]
        print(f"'{sent}->{vector}'")

    print("\n   ⚠️  Limitations:")
    print("      - Ignores word order: 'dog bites man' = 'man bites dog'")
    print("      - Ignores semantics: 'cat' and 'kitten' are unrelated")
    print("      - High dimensional (one dimension per word)\n")

    print("2. Modern Solution: Neural Embeddings")
    print("   - Pre-trained on massive text corpora")
    print("   - Capture semantic relationships")
    print("   - Fixed dimensions (e.g., 384, 768)")
    print("   - Understand context!\n")

    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    load_time = time.time()-start_time
    print(f"Model loaded in {load_time:.2f}seconds\n")
    print(model.state_dict().keys())
    embeddings = model.encode(examples)
    print(f"{embeddings.shape}")
    print(f"  - {embeddings.shape[0]} sentences")
    print(f"  - {embeddings.shape[1]} dimensions each\n")

    print(f" {embeddings[0][:10]}\n")

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)

    print("Cosine similarities between sentences:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = similarities[i][j]
            print(f"  Sentence {i+1} ↔ Sentence {j+1}: {sim:.4f}")
            print(f"    '{sentences[i]}'")
            print(f"    '{sentences[j]}'")
            print()

    print("💡 Notice: Sentences 1 and 2 have HIGH similarity (both about AI/ML)")
    print("   Sentences with weather have LOW similarity to AI/ML sentences!\n")

def deal_with_job_ads():
    corpus_path = Path("example_corpus")
    job_ads = {}
# 代码功能简短总结
# 文件筛选与排序：在corpus_path目录下，筛选出所有以 “job_ad_” 开头、以 “.txt” 结尾的文件，并按顺序排序。
# 文件读取与数据存储：逐个打开筛选后的文件（以 UTF-8 编码读取），将每个文件的内容读取出来，以 “文件名前缀（去除扩展名，如 job_ad_1）” 为键、文件内容为值，存入job_ads字典中。
    for file_path in sorted(corpus_path.glob("job_ad_*.txt")):
        with open(file_path, 'r',encoding = 'utf-8') as f:
            job_ads[file_path.stem] = f.read()
    print(f"Loaded {len(job_ads)}job advertisement")

    for i, (name,content) in enumerate(list(job_ads.items())[:2],1):
        preview = content[:200].replace('\n',' ')
        print(f"{i}.{name}:")
        print(f"  {preview}...\n")
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    print("\nCreating embeddings for all job ads...")
    job_texts = list(job_ads.values())
    job_names = list(job_ads.keys())

    start_time = time.time()

    #需要注意的是，这里的嵌入不同于GPT类型的LLM，他的嵌入是针对句子的，每个句子对应一个384维的向量，模型通过transformer来捕获其语义内涵和上下文信息
    embeddings = model.encode(job_texts,show_progress_bar=True)
    encode_time = time.time() - start_time

    print(
        f"\n✓ Created {len(embeddings)} embeddings in {encode_time:.2f} seconds")
    print(f"  Shape: {embeddings.shape}\n")


    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)

    print("Most similar job ad pairs:")
    pairs = []
    for i in range(len(similarities)):
        for j in range(i+1,len(similarities)):
            pairs.append((i, j, similarities[i][j]))
    pairs.sort(key = lambda x:x[2],reverse=True)
    for i,j,sim in pairs[:3]:
        print(f"\n {job_names[i]} <-> {job_names[j]}")
        print(f"Similarity: {sim:.4f}")
    print("\n💡 High similarity means these jobs require similar skills/experience!\n")

# 定义了语料库路径 corpus_path，指向名为 "example_corpus" 的目录；
# 初始化空字典 job_ads，用于存储职位招聘信息；
# 遍历 corpus_path 目录下、按文件名排序的所有以 "job_ad_" 开头且后缀
# 为 ".txt" 的文件（即职位广告文本文件），文件路径暂存于 file_path 变量（当前代码未体现后续文件内容处理逻辑）
def operation_2():
    
    corpus_path =Path("example_corpus")
    job_ads = {}
    for file_path in sorted(corpus_path.glob("job_ad_*.txt")):
        with open(file_path,'r',encoding='utf-8') as f:
            job_ads[file_path.stem] = f.read()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_names = list(job_ads.keys())
    job_texts = list(job_ads.values())
    embeddings = model.encode(job_texts,show_progress_bar=True)
    print(f"Create {len(embeddings)} embeddings({embeddings.shape[1]}dimensions)")

    def search(query,top_k=3):
        query_embeddings = model.encode(query,show_progress_bar=True)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embeddings],embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        # 这里显式传入第二个参数 1，强制让枚举的第一个元素（最相似的结果）对应 rank=1，因此排名从 1 开始。
        # 二、业务层面的合理性（为什么要这么设计）
        # 这是排名类功能的通用设计习惯：对用户 / 业务侧来说，「第 1 名、第 2 名、第 3 名」是符合人类认知的表述，没人会说「第 0 名」；
        # 你的函数是 search（搜索）功能，返回「top_k=3」的结果，用 1/2/3 标注排名，比 0/1/2 更直观、更符合使用习惯
        for rank, idx in enumerate(top_indices,1):
            results.append({
                'rank':rank,
                'documents':job_names[idx],
                #格式处理：将文本中的换行符（\n）替换为空格，统一文本显示格式，消除换行导致的格式混乱
                'texts':job_texts[idx][:150].replace('\n',' '),
                'similarties': similarities[idx]
            })
        return results

    queries = [
        "Python developer with machine learning experience",
        "Web developer position",
        "Software engineer role"
    ]

    for query in queries:
        results = search(query, top_k=3)

        print(f"Results for: '{query}'")
        print("-" * 80)
        for r in results:
            print(f"\n{r['rank']}. {r['document']}")
            print(f"   Similarity: {r['similarity']:.4f}")
            print(f"   Texts: {r['texts']}...")
        print()

###传统方法：关键词配对，根据配对数排序。现代方法：语义配对，语义通过计算transformer转换的嵌入向量的余弦相似度来检索
### 注意力机制使得语义检索可以理解同义词以及相关概念
###✓ Semantic search: Finds 'software engineer' when searching 'developer'"
###✓ Semantic search: Finds 'software engineer' when searching 'developer'"



if __name__ == "__main__":
    operation()
    deal_with_job_ads()