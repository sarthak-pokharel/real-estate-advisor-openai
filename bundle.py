


import dotenv
dotenv.load_dotenv()



import os
import streamlit as st
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password', value=os.environ['OPENAI_API_KEY'])
TAVILY_API_KEY = st.sidebar.text_input('TAVILY API Key', type='password', value=os.environ['TAVILY_API_KEY'])

os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

if OPENAI_API_KEY=="" or  TAVILY_API_KEY == "":
    exit()




from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel, Field



from enum import Enum

from consultation import consultation_workflow_invoker
from search import search_workflow_invoker, property_stringifyer


class QueryType(BaseModel):
    queryType: str = Field(description="Either of PropertySearchQuery or ConsultationQuery based on user input. ")


llm = ChatOpenAI(temperature=0)
parser = JsonOutputParser(pydantic_object=QueryType)
template = PromptTemplate.from_template(
    "You are a helpful assistant in a real estate company. "
    "Based on the following user input, identify wheather the user wants real estate advice or search for properties in london"
    "{input}"
    "\n{finstr}\n",
    partial_variables=dict(finstr=parser.get_format_instructions())
)
prompt_type_chain = template | llm | parser

def invoke_query(user_inp):
    resp = prompt_type_chain.invoke({'input': user_inp})
    if resp['queryType'] == "PropertySearchQuery":
        return ("PropertySearchQuery")
    else:
        return ("ConsultationQuery")
    pass








def generate_response(input_text):
    resp = invoke_query(input_text)
    if resp == "PropertySearchQuery":
        search_invoc = search_workflow_invoker(input_text)
        wp = search_invoc['winner_property']
        reasoning = search_invoc['winner_reasoning']
        st.write("[{resp}] The best property calculated was \n")
        st.write(f"{property_stringifyer(wp)}.\n\n")
        st.info(f"\n{reasoning}")
        st.info(f"\n{wp.url}")
        return
    consul_invoc = consultation_workflow_invoker(input_text)
    outp = f"[{resp}]\n"+consul_invoc
    st.info(outp)
    return


with st.form('my_form'):
    
    text = st.text_area('Enter Query:', '')
    if st.form_submit_button('Submit'):
        generate_response(text)
    pass
