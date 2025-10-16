from ibm_watsonx_ai import Credentials
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_core.documents import Document

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Type
from IPython.display import display, JSON, Markdown
from crewai.tools import BaseTool
from langchain.tools import tool

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

import yaml
import os 
import json

# Define credentials for LLMs usage
credentials = Credentials(
                url = "https://us-south.ml.cloud.ibm.com",
                )

project_id = "skills-network"

class RetrieverToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    user_profile: str = Field(..., description="The string version of the user profile dictionary")

class UserProfile(BaseModel):
    preferred_cuisines: Dict[str, float] = Field(..., description="A dictionary of cuisines and a preference score (0-1) for each.")
    price_tier_preference: int = Field(..., description="The user's preferred price tier (e.g., 0 for the lowest, 1 for $, 2 for $$, etc.).")
    avg_rating_preference: float = Field(..., description="The average rating the user prefers in restaurants (e.g., 4.5/5).")
    dining_environment_preference: str = Field(..., description="A short summary of the user's preferred dining environment.")
    summary: str = Field(..., description="A short text summary of the overall user profile.")

def crew_builup():

    path = 'restaurant-database/Synthetic-Restaurants-Cafes-Bakeries.json' # Database URL

    # Check if the request was successful (status code 200)
    with open(path, 'r') as f:
        restaurant_db = json.load(f)

    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=credentials["url"],
        project_id=project_id,
        )

    doc_id = 0
    documents = []
    sources = list(restaurant_db.keys()) #restaurant or cafe or bakery
    for source in sources:
        for j in range(len(restaurant_db[source])):
            shop = restaurant_db[source][j]

            signature_dishes = ''
            for i in range(len(shop['signature_items'])):
                signature_dishes += f"Dish {i+1}: {shop['signature_items'][i]}.\n"
        
            review_contents = ''
            for i in range(len(shop['review_titles'])):
                review_contents += f"Review {i+1}. Title: {shop['review_titles'][i]}. Text: {shop['reviews'][i]}.\n"
                
            content = (
                f"Shop name: {shop['label']}. " + 
                f"Shop type:  {shop['type']}. " + 
                f"Shop location: {shop['location']}. " +
                f"Shop rating: {shop['rating']}. " +
                f"Shop price_range: {len(shop['price_range'])//3}. " +
                f"Shop short description: {shop['short_description']}. \n\n" +
                "The following are the signature dishes. \n" +
                signature_dishes + '\n' +
                'The following are the sampled reviews. \n' +
                review_contents
            )
            document = Document(
                page_content=content,
                metadata={"source": source, "id":doc_id},
                id=doc_id,
            )
            doc_id += 1
            documents.append(document)

    vector_db = Chroma(
        collection_name='Database',
        embedding_function=embeddings,
    )

    vector_db.add_documents(documents=documents)

    retriever = vector_db.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    class RAG_Retriever(BaseTool):
        name: str = "RAG Retriever"
        description: str = "Coarse top 10 recommendations"
        args_schema: Type[BaseModel] = RetrieverToolInput

        def _run(self, user_profile: str) -> str:
            # Your tool's logic here
            res = retriever.invoke(user_profile)
            recommendation_dict = {}
            for i in range(len(res)):
                recommendation_dict[f'Recommendation {i+1}'] = res[i].page_content
            return recommendation_dict
            
    retriever_tool = RAG_Retriever()

    llm = LLM(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
            base_url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            max_tokens=2000,
    )

    agents_config_path = 'agent-src/agent.yaml'
    tasks_config_path = 'agent-src/tasks.yaml'

    with open(agents_config_path, 'r') as f:
        agents_config = yaml.safe_load(f)
        
    with open(tasks_config_path, 'r') as f:
        tasks_config = yaml.safe_load(f)

    user_profile_builder = Agent(
        role=agents_config['user_profile_agent']['role'],
        goal=agents_config['user_profile_agent']['goal'],
        backstory=agents_config['user_profile_agent']['backstory'],
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    coarse_RAG_matcher = Agent(
        role=agents_config['coarse_RAG_matcher']['role'],
        goal=agents_config['coarse_RAG_matcher']['goal'],
        backstory=agents_config['coarse_RAG_matcher']['backstory'],
        tools=[retriever_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    restaurant_recommendation_expert = Agent(
        role=agents_config['restaurant_recommendation_agent']['role'],
        goal=agents_config['restaurant_recommendation_agent']['goal'],
        backstory=agents_config['restaurant_recommendation_agent']['backstory'],
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    food_trend_researcher = None

    user_profile_task = Task(
        description=tasks_config['user_profile_task']['description'],
        expected_output=tasks_config['user_profile_task']['expected_output'],
        agent=user_profile_builder,
        output_pydantic=UserProfile
    )

    coarse_RAG_match_task = Task(
        description=tasks_config['coarse_RAG_match_task']['description'],
        expected_output=tasks_config['coarse_RAG_match_task']['expected_output'],
        agent=coarse_RAG_matcher,
        depends_on=['user_profile_task'],
        input_data=lambda outputs:{
            'user_profile': outputs['user_profile_task'].raw
        }
    )

    food_trend_task = None

    if food_trend_task is not None:
        restaurant_recommendation_task = Task(
            description=tasks_config['restaurant_recommendation_task']['description'],
            expected_output=tasks_config['restaurant_recommendation_task']['expected_output'],
            agent=food_trend_researcher,
            depends_on=['user_profile_task','coarse_recommend_task','food_trend_task'],
            input_data=lambda outputs:{
                'user_profile': outputs['user_profile_task'].raw,
                'database_restaurants': outputs['coarse_recommend_task'].raw,
                'trending_restaurants': outputs['food_trend_task'].raw
            }
        )
    else:
        restaurant_recommendation_task = Task(
            description=tasks_config['restaurant_recommendation_task']['description'],
            expected_output=tasks_config['restaurant_recommendation_task']['expected_output'],
            agent=food_trend_researcher,
            depends_on=['user_profile_task','coarse_recommend_task','food_trend_task'],
            input_data=lambda outputs:{
                'user_profile': outputs['user_profile_task'].raw,
                'database_restaurants': outputs['coarse_recommend_task'].raw
            }
        )

    if food_trend_researcher is not None:
        crew = Crew(
            agents=[user_profile_builder,coarse_RAG_matcher,food_trend_researcher,restaurant_recommendation_expert],
            tasks=[user_profile_task,coarse_RAG_match_task,food_trend_task,restaurant_recommendation_task],
            process=Process.sequential,
            verbose=False,
            output_log_file='crew_log.txt'
        )
    else:
        crew = Crew(
            agents=[user_profile_builder,coarse_RAG_matcher,restaurant_recommendation_expert],
            tasks=[user_profile_task,coarse_RAG_match_task,restaurant_recommendation_task],
            process=Process.sequential,
            verbose=False,
            output_log_file='crew_log.txt'
        )

    return crew