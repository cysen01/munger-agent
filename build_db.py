import os
import json
import shutil
# 🔧 修正点：使用新版 import 路径，或者自动回退
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- ⚙️ 核心配置 ---
DB_PATH = "./chroma_db"
JSON_PATH = "book_chunks.json"  # 👈 你的黄金数据

def main():
    print("🚀 开始构建查理·芒格知识库 (JSON修复版)...")

    # ==========================================
    # 🧹 步骤 1：自动清理旧数据
    # ==========================================
    if os.path.exists(DB_PATH):
        print(f"🧹 检测到旧数据库 {DB_PATH}，正在自动删除...")
        try:
            shutil.rmtree(DB_PATH)
            print("✅ 旧数据清理完毕。")
        except Exception as e:
            print(f"⚠️ 清理失败，请先关闭 app.py！错误: {e}")
            return

    # ==========================================
    # 📖 步骤 2：读取 JSON
    # ==========================================
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误：找不到 {JSON_PATH}！请检查文件名。")
        return

    print(f"📖 正在读取 {JSON_PATH}...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 格式错误: {e}")
        return
    
    print(f"📄 成功加载，原始数据包含 {len(data)} 个块")

    # ==========================================
    # 🔄 步骤 3：转换数据
    # ==========================================
    print("🔄 正在打包数据...")
    documents = []
    for item in data:
        # 兼容 content 或 text 字段
        content = item.get("content") or item.get("text")
        if content:
            meta = {
                "source": "book_chunks.json", 
                "chunk_id": item.get("chunk_id"),
                "length": item.get("length")
            }
            doc = Document(page_content=content, metadata=meta)
            documents.append(doc)
    
    if not documents:
        print("❌ 警告：JSON 里没读到有效内容！")
        return

    print(f"📦 准备入库 {len(documents)} 个精华片段")

    # ==========================================
    # 🧠 步骤 4：入库
    # ==========================================
    print("🧠 正在初始化向量模型...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"💾 正在写入数据库...")
    vector_db = Chroma.from_documents(
        documents=documents, 
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"✅ 恭喜！知识库构建完成！共存入 {len(documents)} 条数据。")

if __name__ == "__main__":
    main()