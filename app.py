__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from openai import OpenAI
import os
import time
import build_db  # 自动初始化脚本

# ==========================================
# 🔧 配置区域 (云端安全版)
# ==========================================
st.set_page_config(page_title="查理·芒格：普世智慧 (导师版)", page_icon="👴", layout="wide")

# 🔒 安全修改：优先从 Secrets 获取 Key
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    # 本地运行时的备用 Key (部署前请确保这里不要留真实的 Key，或者只在本地测试用)
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

st.title("👴 查理·芒格：普世智慧 (博学导师版)")

# ==========================================
# 🧠 核心逻辑
# ==========================================

@st.cache_resource
def load_resources():
    # 自动构建逻辑：云端第一次运行时，如果没有库，自动从 JSON 构建
    if not os.path.exists("./chroma_db"):
        st.warning("🚀 云端首次运行，正在构建知识库... (约需1分钟)")
        try:
            build_db.main()
            st.success("✅ 构建完成！")
        except Exception as e:
            st.error(f"构建失败: {e}")
            return None, None

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return db, reranker

@st.cache_resource
def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

try:
    with st.spinner("🚀 正在连接芒格大脑..."):
        vector_db, reranker_model = load_resources()
        client = get_client()
    if vector_db:
        st.toast("✅ 系统已就绪", icon="🧠")
except Exception as e:
    st.error(f"系统初始化失败: {e}")

# ==========================================
# ⚡ 侧边栏逻辑 (带 Session State)
# ==========================================
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {"status": "等待提问...", "top_docs": []}

with st.sidebar:
    st.header("🧠 思维链监控")
    st.info(st.session_state.debug_info["status"])
    
    if st.session_state.debug_info["top_docs"]:
        st.divider()
        st.write("**🏆 当前参考片段:**")
        for i, (doc, score) in enumerate(st.session_state.debug_info["top_docs"]):
            st.success(f"Top {i+1} | 权重: {score:.2f}")
            st.caption(doc.page_content[:100] + "...")
    
    st.divider()
    if st.button("🗑️ 清空历史"):
        st.session_state.messages = []
        st.session_state.debug_info = {"status": "等待提问...", "top_docs": []}
        st.rerun()

# ==========================================
# 💬 聊天界面
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请向芒格先生提问..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if vector_db and reranker_model:
        # 清空侧边栏状态
        st.session_state.debug_info["status"] = "⏳ 正在检索新数据..."
        st.session_state.debug_info["top_docs"] = []
        
        with st.status("👴 芒格正在调用多元思维模型...", expanded=True) as status:
            st.write("🔍 正在检索《穷查理宝典》...")
            raw_docs = vector_db.similarity_search(prompt, k=30)
            
            # 去重
            seen_content = set()
            unique_docs = []
            for doc in raw_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            initial_docs = unique_docs[:20]
            
            st.write("⚖️ 正在进行深度价值评估 (Rerank)...")
            pairs = [[prompt, doc.page_content] for doc in initial_docs]
            scores = reranker_model.predict(pairs)
            scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:5]]
            
            # 更新侧边栏
            st.session_state.debug_info["status"] = "✅ 检索完成"
            st.session_state.debug_info["top_docs"] = scored_docs[:5]
            
            time.sleep(0.2)
            status.update(label="👴 思考完成，准备输出智慧。", state="complete", expanded=False)

        context_text = "\n".join([f"- {doc.page_content}" for doc in top_docs])

        # ============================================================
        # 🏆 你的新版 Prompt (博学导师版)
        # ============================================================
        system_prompt = """
        你现在是查理·芒格 (Charlie Munger) 的数字意识。
        
        【角色定位】：
        你不是一个只会骂人的怪老头，而是一位**博学、严谨、虽然毒舌但充满关怀的老师**。
        你看到年轻人犯错时，不会只是冷笑一声走开，而是会**停下来，用你的智慧（思维模型）把他的错误拆解给他看**，让他心服口服。
        
        【回答风格】：
        1. **拒绝敷衍**：不要只给一句话的结论。要解释**“为什么”**。
        2. **多元思维模型**：回答问题时，必须显式或隐式地调用多个学科的知识（心理学、数学、工程学、历史）。
        3. **深度解析**：不要只说“这是愚蠢的”，要说“这之所以愚蠢，是因为你忽视了复利效应/误判了概率/掉进了社会认同的陷阱”。
        4. **引用历史/案例**：芒格非常喜欢引用历史故事（如罗马帝国的衰落、李光耀的治理、富兰克林的名言）来佐证观点。
        5. **回答减少框架感，像一位智慧的长者娓娓道来，层层深入，不要弄成一个提纲
        6. **不要一直用年轻人开头
        
        【禁忌】：
        - 🚫 禁止列干巴巴的提纲（第一、第二...）。要像写文章或演讲一样自然流畅。
        - 🚫 禁止使用“(冷笑)”等舞台剧动作。
        - 🚫 禁止无理由的辱骂。你的傲慢来自于智力上的降维打击，而不是脏话。

        【特殊场景】：
        - 如果用户问“怎么做”，不要给操作手册，要给“原则”。
        - 如果用户问礼貌的废话（“你好”），礼貌但简短地回应，并引导他问有价值的问题。
        """
        
        full_user_prompt = f"【参考资料】:\n{context_text}\n\n【用户的问题】:\n{prompt}\n\n【要求】：请像查理·芒格在股东大会上那样，深入浅出地剖析这个问题。"

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt},
                ],
                stream=True
            )
            response = st.write_stream(stream)
        

        st.session_state.messages.append({"role": "assistant", "content": response})
