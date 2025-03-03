import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="ICoE GPT | Chat-Bot 🤖")


#Contact
with st.sidebar.expander("📬 Contact"):

    st.write("**GitHub:**",
"[yvann-hub/ICoE GPT-chatbot](https://github.com/D161764/icoegpt)")



    st.write("**Twitter:** [@sudiphcp](https://twitter.com/sudiphcp)")
    st.write("**Mail** : sudip.architect@gmail.com")
    st.write("**Created by D161764**")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>ICoE GPT, your Integration Log Analyser 🤖</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>I'm ICoE GPT, an intelligent chatbot created by combining 
    the strengths of Langchain and Streamlit. I use large language models to provide
    context-sensitive interactions. My goal is to help you analyse Integration Log.
    I support PDF, TXT, CSV, Youtube transcript 🧠</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")


#ICoE GPT's Pages
st.subheader("🚀 ICoE GPT's Pages")
st.write("""
- **ICoE GPT-Chat**: General Chat on data (PDF, TXT,CSV) with a [vectorstore](https://github.com/facebookresearch/faiss) (index useful parts(max 4) for respond to the user) | works with [ConversationalRetrievalChain](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html)
- **ICoE GPT-Sheet** (beta): Chat on tabular data (CSV) | for precise information | process the whole file | works with [CSV_Agent](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/csv.html) + [PandasAI](https://github.com/gventuri/pandas-ai) for data manipulation and graph creation
- **ICoE GPT-Youtube**: Summarize YouTube videos with [summarize-chain](https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html)
""")
st.markdown("---")


#Contributing
st.markdown("### 🎯 Contributing")
st.markdown("""
**ICoE GPT is under regular development. Feel free to contribute and help me make it even more data-aware!**
""", unsafe_allow_html=True)





