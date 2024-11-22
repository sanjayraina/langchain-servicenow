import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import re

import requests
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from servicenow import ServiceNowLoader
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

load_dotenv()
os.environ['SERVICENOW_FILTER'] = "active=true"
os.environ['SERVICENOW_FIELDS'] = "number,short_description,description"
IncidentFields = ["number", "short_description",
                  "state", "priority", "opened_at"]

from snow_api_wrapper import ServiceNowAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from langchain.agents import create_openai_functions_agent, Tool
from langchain import hub
from langchain.agents import AgentExecutor

MODEL = "gpt-3.5-turbo"
QUESTION_HISTORY: str = 'question_history'
SERVICENOW_GET_RECORDS_PROMPT = """
    This tool is a wrapper around the ServiceNow table API, useful when you need to search for records in ServiceNow.
    Records can be of type Incident, Change Requests or Service Requests.
    The input to the too is a query string that filters results based on attribute value.
    For example, to find all the Incident records with Priority value of 1, you would 
    pass the following string: priority=1
    """
SERVICENOW_INC_CREATE_PROMPT = """
    This tool is a wrapper around the ServiceNow incident table API, useful when you need to create a ServiceNow incident. 
    The input to this tool is a dictionary specifying the fields of the ServiceNow incident, and will be passed to the table API call.
    For example, to create an incident called "test issue" with description "test description", you would pass in the following dictionary: 
    {{"short_description": "test issue", "description": "test description"}}
    """
PLOT_PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "If the query asks to plot data then generate the code in plotly."
    "The solution should be given using plotly and only plotly. Do not use matplotlib."
    "Return the code <code> in the following format ```python <code>```"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)

USECASES = ["ServiceNow Agent", "ServiceNow RAG"]
TABLES = ["kb_knowledge", "incident", "problem", "change_request"]


def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        code = matches[0]
        code = code.replace("fig.show()", "st.plotly_chart(fig, theme='streamlit', use_container_width=True)")
        # code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
        return code
    
# Main()
#
st.set_page_config(
    page_title="LangChain-ServiceNow",
    layout="wide",
    page_icon="🦜✚🛠️",
)

st.write("# Langchain + ServiceNow Examples ")

st.markdown(
    """
    These examples show how LLMs can be combined with workflow systems to 
    build intelligent business process orchestration capabilities
"""
)

use_case = st.sidebar.selectbox('Choose Use Case', USECASES)
table = st.sidebar.selectbox('Choose table', TABLES)


match (use_case):
    # RAG use case
    #
    case 'ServiceNow RAG':
        with st.sidebar:
            query = st.text_input("Filter", value="short_descriptionLIKEsmartcool")
            fields = st.text_input("Fields", value="short_description,text")
        # query = "short_descriptionLIKEsmartcool"
        # fields = "short_description,text"

        # Call ServiceNow Doc Loader to ingest data
        loader = ServiceNowLoader(table, query, fields)
        docs = loader.lazy_load()
        doc_data = []
        for doc in docs:
            doc_data.append(doc)
        
        # Use embeddings model to store document embeddings
        embeddings = OpenAIEmbeddings()
        embedding_list = embeddings.embed_documents(
                    [text.page_content for text in doc_data])
        vectordb = DocArrayInMemorySearch.from_documents(doc_data, embeddings)

        # Set up Retriever and chain
        retriever = vectordb.as_retriever()
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    output_key='answer'
                )
        chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    verbose=True,
                    max_tokens_limit=4000
                )
        
        if st.sidebar.button("Clear message history"):
            memory.chat_memory.clear()
            avatars = {"human": "user", "ai": "assistant"}

        if len(memory.chat_memory.messages) == 0:
            st.chat_message("assistant").markdown("Ask me a question about ServiceNow KB!")
        else :
            for msg in memory.chat_memory.messages:
                st.chat_message(avatars[msg.type]).write(msg.content)

        # assistant = st.chat_message("assistant")
        
        if user_query := st.chat_input(placeholder="Enter your query here."):
            st.chat_message("user").write(user_query)
            container = st.empty()
            stream_handler = StreamlitCallbackHandler(container)
            with st.chat_message("assistant"):
                response = chain.run({
                    "question": user_query,
                    "chat_history": memory.chat_memory.messages,
                },
                callbacks=[stream_handler])
                # Display the response from the chatbot
                if response:
                    container.markdown(response)

    #  Agent use case
    #
    case 'ServiceNow Agent':
        chat_type = st.sidebar.radio("What type of chat do you want?", ["Action", "Analysis"])

        c1 = st.container(border=True)          # c1 container displays the list of records in AG Grid
        c2 = st.container(border=True)          # c1 container displays the chat interface
        with c1:
            snow_url = os.environ['SERVICENOW_INSTANCE_URL']
            snow_username = os.environ['SERVICENOW_INSTANCE_USERNAME']
            snow_password = os.environ['SERVICENOW_INSTANCE_PASSWORD']

            if (not snow_url):
                c1.write("ServiceNow Env variable not setup")
            else:
                url = snow_url + "/api/now/table/" + table + "?sysparm_limit=100"

            headers = {"Content-Type": "application/json",
                    "Accept": "application/json"}
            response = requests.get(url, auth=(snow_username, snow_password), headers=headers)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data['result'])
                df = df.reset_index()
                df['Id'] = df['index'].astype(str)
                df = df.drop(columns=['index'])
                # data = df.to_dict('records')
                df_list = df[IncidentFields]
                # st.write(df.head(1))
                gb = GridOptionsBuilder.from_dataframe(df_list)
                gb.configure_pagination()
                gb.configure_side_bar()
                gb.configure_selection("single")
                # gb.configure_column('number', editable=False)
                gridOptions = gb.build()
                ag = AgGrid(df_list, key=None, gridOptions=gridOptions)
                sel_row = ag['selected_rows']

            if (chat_type == "Action"):
                servicenow = ServiceNowAPIWrapper()                 # ServiceNow API tool
                duckduck_search = DuckDuckGoSearchAPIWrapper()      # Web search tool

                tools = [
                    Tool.from_function(
                        name = "ServiceNow_Create",
                        func=servicenow.create_incident,
                        description=SERVICENOW_INC_CREATE_PROMPT
                    ),
                    Tool(
                        name = "Search_DuckDuckGo",
                        func=duckduck_search.run,
                        description="useful for when you need to answer questions about current events. You should ask targeted questions"
                    )
                ]

                llm = ChatOpenAI(model=MODEL, temperature=0)
            
                prompt = hub.pull("hwchase17/openai-functions-agent")
                agent = create_openai_functions_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                if isinstance(sel_row, pd.DataFrame):     
                    num = ag.selected_rows.number.iloc[0]
                    df_record = df.loc[df['number'] == num]    #  Pull the record to query on

                    user_question = c2.text_input("Ask a question about the selected data item")
                    with st.spinner('Please wait ...'):
                        try:
                            agent_input = user_question + "\nIncident Description:\n" + df_record.description.iloc[0]
                            st_callback = StreamlitCallbackHandler(c2.container())
                            response = agent_executor.invoke({"input": agent_input}, {"callbacks": [st_callback]})
                        except Exception as e:
                            st.error(f"Error occurred: {e}")
                else:
                        st.write ("Select a row from table above")
            else:
                # Pandas Dataframe agent
                llm = ChatOpenAI(model=MODEL, temperature=0, max_tokens=256)
                agent = create_pandas_dataframe_agent(llm, df, verbose=True, 
                                        allow_dangerous_code=True, 
                                        agent_type=AgentType.OPENAI_FUNCTIONS)
                
                user_question = c2.text_input("Ask a question to analyse the above data")
                
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.invoke(PLOT_PROMPT + user_question, {"callbacks": [st_callback]})

                code = extract_python_code(response['output'])
                exec(code)
                # c2.write(f"```{code}")


