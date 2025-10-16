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

import os 

import requests
import json

# Define credentials for LLMs usage
credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )

project_id = "skills-network"

path = 'restaurant-database/Synthetic-Restaurants-Cafes-Bakeries.json' # Database URL

# Check if the request was successful (status code 200)
import json
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
            signature_dishes += f'Dish {i+1}: {shop['signature_items'][i]}.\n'
    
        review_contents = ''
        for i in range(len(shop['review_titles'])):
            review_contents += f'Review {i+1}. Title: {shop['review_titles'][i]}. Text: {shop['reviews'][i]}.\n'
            
        content = (
            f'Shop name: {shop['label']}. ' + 
            f'Shop type:  {shop['type']}. ' + 
            f'Shop location: {shop['location']}. ' +
            f'Shop rating: {shop['rating']}. ' +
            f'Shop price_range: {len(shop['price_range'])//3}. ' +
            f'Shop short description: {shop['short_description']}. \n\n' +
            'The following are the signature dishes. \n' +
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