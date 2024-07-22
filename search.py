

# import dotenv
# dotenv.load_dotenv()


import difflib
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from consultation import consultation_app, consultation_workflow_invoker
import concurrent.futures
from typing import List, Optional
from pydantic_core import from_json
from pydantic import BaseModel, Field
from typing import List
import json
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from typing import TypedDict

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tavily_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper())
available_tools = [tavily_tool, wikipedia]



llm = ChatOpenAI(temperature=0)



properties_path = "datasets/apify_rightmove_london_rental.pkl"
import pickle
def load_properties():
    return pickle.load(open(properties_path, 'rb'))
properties = load_properties()


class Station(BaseModel):
    name:str
    distance: float
    unit: str
class Property(BaseModel):
    id: int = Field(description="Unique identifier for property")
    url: str = Field(description="URL of property sale")
    title: str = Field(description="Property Title")
    displayAddress: str = Field(description="Display Address")
    bathrooms: int = Field(description="Num of bathrooms")
    bedrooms: int = Field(description="Num of bedrooms")
    propertyType: str = Field(description="Type of property")
    price: int = Field(description="price in euro per month")
    features: str = Field(description="key features of property")
    description: str = Field(description="Property Description")
    nearestStations: List[Station] = Field(description="Nearby Stations Info")
    borough: str = Field(description="Borough where the property lies")
    ward: str = Field(description="Ward where the property lies")
def PropertyStringify(prop):
    pass
obj_props = [Property.model_validate(from_json(json.dumps(w))) for w in properties]

class NumRange(BaseModel):
    start_from: int = Field(description="Range Starting From Value")
    end_in: int = Field(description="Range Ends To Value")

class FloatRange(BaseModel):
    start_from: float = Field(description="Range Starting From Value")
    end_in: float = Field(description="Range Ends To Value")

class RequiredFields(BaseModel):
    property_description: str = Field(description="Property Description")
    price_range: NumRange = Field(description="Price range. Default Value is 0 to 99999999. ")
    borough: str = Field(description="Borough where property lies or empty if not specified")
    ward: str = Field(description="Ward where property lies or empty if not specified")
    bathrooms_range: NumRange = Field(description="No of bathrooms range. 1 - 999 is the default value. ")
    bedrooms_range: NumRange = Field(description="No of bedrooms range. 1 - 999 is the default value. ")
    special_qualities: str = Field(description="Any special demands of user. Leave empty if not. ")
    nearest_station_distance: FloatRange = Field("Range of distance for for nearest station in miles. 0 to 99 default value.  ")

filter_parser = PydanticOutputParser(pydantic_object=RequiredFields)

filter_prompt = PromptTemplate.from_template(
    (
        "Based on the user query, extract the useful information to filter out properties from our database. \n"
        "User Query: `{query}`\n"
        "{format_instructions}"
    ),
    partial_variables=dict(format_instructions=filter_parser.get_format_instructions())
)


filter_chain = filter_prompt | llm | filter_parser


def generate_filter(state):
    fl = filter_chain.invoke(dict(query=state['search_query']))
    return dict(requirements_filter=fl)


def filter_properties(state):
    
    criteria = state['requirements_filter']
    filtered_properties = []
    
    for prop in obj_props:
        # Check borough
        if criteria.borough and prop.borough != criteria.borough:
            continue
        
        # # Check ward
        # if criteria.ward and prop.ward != criteria.ward:
        #     continue
        
        # Check price range
        if not (criteria.price_range.start_from <= prop.price <= criteria.price_range.end_in):
            continue
        
        # Check bathrooms range
        if not (criteria.bathrooms_range.start_from <= prop.bathrooms <= criteria.bathrooms_range.end_in):
            continue
        
        # Check bedrooms range
        if not (criteria.bedrooms_range.start_from <= prop.bedrooms <= criteria.bedrooms_range.end_in):
            continue
        
        # Check nearest station distance if criteria are specified
        if criteria.nearest_station_distance:
            valid_distance = any(
                station.distance <= criteria.nearest_station_distance.end_in and 
                station.distance >= criteria.nearest_station_distance.start_from
                for station in prop.nearestStations
            )
            if not valid_distance:
                continue
        
        # If all checks passed, add the property to the filtered list
        filtered_properties.append(prop)
    
    state['filtered_properties'] = filtered_properties
    return state


def sort_properties(state):
    sortedV = sorted(
        state['filtered_properties'], 
        key=
            lambda z: 
                    difflib.SequenceMatcher(
                        None, 
                        z.features+" "+z.description, 
                        state['requirements_filter'].special_qualities
                            +
                        state['requirements_filter'].property_description
                    ).ratio(), 
                reverse=True
            )
    
    state['filtered_properties'] = sortedV[:4]
    return state


def property_stringifyer(p):
    # return f"id: {p.id}\ntitle: {p.title}\ndescription: {p.description}\nfeatures: {p.features}\nlocation: {p.displayAddress}\nPrice in Euro: {p.price} \nBathrooms: {p.bathrooms} \nBedrooms: {p.bedrooms} "
    return json.dumps(dict(id=p.id, title=p.title, description=p.description, features=p.features, location=p.displayAddress, price_in_euro=p.price, bathrooms=p.bathrooms, bedrooms=p.bedrooms))



class ID_Extraction(BaseModel):
    unique_id: int
    reasoning: str
id_extraction_parser = JsonOutputParser(pydantic_object=ID_Extraction)
prompt = PromptTemplate.from_template(
    "From a piece of text below, extract unique id of best property and reasoning for suggesting that. "
    "\n```{outp}\n```\n"
    "{format_instructions}",
    partial_variables=dict(format_instructions=id_extraction_parser.get_format_instructions()
))
id_extraction_chain = prompt | llm | id_extraction_parser

def compare_properties(pair):
    prop1, prop2, state = pair
    req_filter = state['requirements_filter']
    response = consultation_workflow_invoker(
        property_stringifyer(prop1)
        +
        "\n"
        +
        property_stringifyer(prop2)
        +
        "\n\n"
        +
        f"The user's query is following: {str(req_filter)}. "
        "Out of above two properties, help me out finding which is the better one for user. Make sure to give me the unique id of better one.  \n"
    )
    w = id_extraction_chain.invoke(dict(outp=response))
    winner_id = w['unique_id']
    winner_reasoning = w['reasoning']
    state['winner_reasoning'] = winner_reasoning
    return prop1 if int(prop1.id) == winner_id else prop2



def parallel_compare_pairs(array, state):
    # Create pairs from the array
    pairs = [(array[i], array[i + 1], state) for i in range(0, len(array) - 1, 2)]
    
    # Use ThreadPoolExecutor to parallelize the comparison
    with concurrent.futures.ThreadPoolExecutor() as executor:
        winners = list(executor.map(compare_properties, pairs))
    
    return winners

def find_winner(arr, compare_fn, state):
    while len(arr) > 1:
        # If the length is odd, append the last element as is
        if len(arr) % 2 != 0:
            arr.append(arr[-1])
        
        arr = parallel_compare_pairs(arr, state)
    
    return arr[0] if arr else None

def winner_property_node(state):
    filtered_props_torace = state['filtered_properties'].copy()
    winner = find_winner(filtered_props_torace, compare_properties, state=state)
    state['winner_property'] = winner
    return state


class SearchState(TypedDict):
    search_query: str
    requirements_filter: RequiredFields
    filtered_properties: List
    winner_property: Property
    winner_reasoning: str

workflow = StateGraph(SearchState)

workflow.add_node("generate_filter", generate_filter)
workflow.add_edge(START, "generate_filter")

workflow.add_node("filter_properties", filter_properties)
workflow.add_edge("generate_filter","filter_properties")


workflow.add_node("sort_properties", sort_properties)
workflow.add_edge("filter_properties", "sort_properties")


workflow.add_node("winner_property_node", winner_property_node)
workflow.add_edge("sort_properties", "winner_property_node")

workflow.add_edge("winner_property_node", END)

search_app = workflow.compile()





def search_workflow_invoker(query):
    res = search_app.invoke(dict(
        search_query=query,
        filtered_properties=[]
    ))
    return res


# outp = search_workflow_invoker("I need an apartment,  price below 3000 euros per month, in westminister in london. it should be beautifully well furnished.")
# print(outp)