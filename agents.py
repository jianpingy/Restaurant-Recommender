import yaml
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

import os 

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