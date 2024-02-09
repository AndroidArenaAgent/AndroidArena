import itertools
import os
import traceback
import logging

import numpy as np
from dotenv import load_dotenv
from langchain import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser, ListOutputParser

import re

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import AsyncHtmlLoader, WikipediaLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseRetriever, Document, BaseDocumentTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

load_dotenv(".env")


class SearchQueries(BaseModel):
    """Search queries to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )


DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with generating Google Search queries where their searched results can cover all functions and basic usage instructions of the given APPs. 
    For example, for the give <APP>, the generated queries should be like 'how to use <APP>', '<APP> guidance', '<APP> usage instruction' etc.
Generate the Google search queries as many and diverse as possible. The output should be a numbered list of questions: {question}""",
)

TASK_SEED_TEMPLATE_PROMPT = PromptTemplate(
    input_variables=["feature"],
    template="""You are a smart task creator, where instructions can be generated based on these templates. For example, we can generate "create an event titled 'team meeting' for 3PM" and "create an event titled 'go to the hospital for 11AM" based on the template "create an event titled <event title> for <event time>". Your goal is to generate tasks templates for automatic features from the feature description of an APP:

{feature}

Please generate as many of these task templates as possible for the app. Your response should be a numbered list of task templates.""",
)

TASK_SEED_PROMPT = PromptTemplate(
    input_variables=["feature", "app"],
    template="""You are a smart task creator for a smartphone intelligent assistant. Given the features description of the {app} APP, your goal is to generate clear and practical tasks that the assistant can assist people with while they use {app} on their phone in their daily lives. These tasks should encompass a wide range of possible instructions and questions that may arise when using {app} APP.

For example, for the Gmail APP, potential task instructions could include:
Compose an email with the subject <email subject> and the message content <email content> to be sent to <email address> using Gmail., 
Send the first draft email., 
Open the latest email from <email address> in Gmail., 
Open Gmail settings., 
Turn off notifications for Gmail., 
Star the latest email from <email address> in Gmail., 
Delete the latest email from <email address> in Gmail., 
etc., where the placeholders surrounded with angle brackets '<' and '>' should be automated generated and not be filled with specific content.

The {app} APP's feature description is: 
{feature}

Your task now is to generate as many of these tasks as possible for the {app} app. Ensure that these instructions are clear and will not lead to any misunderstanding so that the assitant can successfully execute them.
Your response should be a list of comma separated task instructions, where each instruction should be presented in one sentence.""",
)

CROSS_TASK_SEED_PROMPT = PromptTemplate(
    input_variables=["feature", "app"],
    template="""You are a proficient task creator for a smartphone's intelligent assistant. Your objective is to craft explicit and practical cross-APP tasks that can be cooperatively accomplished by the {app} APPs, leveraging the feature descriptions of these apps. These tasks should encompass a wide array of potential instructions and questions that might arise in users' daily lives when utilizing {app} on their smartphones.

For example, for the Gmail and Google Calendar APPs, potential cross-APP task instructions could include:
Find the email with the subject <subject> in your Gmail, extract the meeting details, and create an event in Google Calendar., 
Search Gmail for the latest email related to upcoming flights, extract the flight details, and create a calendar event for the flight in Google Calendar., 
Scan Gmail for the latest event invitation and RSVP confirmations, and automatically update Google Calendar with the RSVP status for the event., 
etc., where the placeholders surrounded with angle brackets '<' and '>' should be automated generated and not be filled with specific content.

The {app} APPs‘ features and functions description are: 
{feature}

Your task now is to generate as many of these cross-APP tasks as possible for the {app} APPs. 
Ensure that the generated cross-APP tasks must be cooperatively completed by the {app} APPs, and these instructions should be clear, comprehensive, and free from ambiguity to enable the assistant to execute them successfully. 
Your response should be a list of comma separated task instructions, where each instruction MUST be presented in one line of sentence.""",
)

POST_PROMPT = """Please note that the #Given Instruction# might be a template with placeholders surrounded with angle brackets '<' and '>', e.g., 'Compose an email with the subject <email subject> and the message content <email content> to be sent to <email address> using Gmail.'. You should fill the placeholders with specific content and generate a pratical instruction, e.g., ’Compose an email with the subject "Hello" and the message content "Hello, world!" to be sent to abc@example.com using Gmail.‘.
Ensure that the #New Instruction# remains a practical and realistic {app} APP task instruction for a mobile phone user, but do not incorporate personal information.
Concisely and accurately output the generated instruction in one line.
#Given Instruction#:
{instruction}

#APP's functionality#:
{feature}

The #New Instruction# is:
"""

ADD_CONSTRAINTS_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone. 
Please add a few more constraints or requirements to #Given Instruction#, and create #New Instruction#.
""" + POST_PROMPT
)

COMPLICATE_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone. 
    Please rewrite #Given Instruction# to make it slightly more complicated, and create #New Instruction#.
""" + POST_PROMPT
)

DEEPEN_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone. 
    Slightly increase the depth and breadth of #Given Instruction#, and create #New Instruction#.
""" + POST_PROMPT
)

CONCRETIZE_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone.  
    Make #Given Instruction# slightly more concrete, and create #New Instruction#.
""" + POST_PROMPT
)

INCREASE_REASONING_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone. 
    If #Given Instruction# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Instruction#.
""" + POST_PROMPT
)

SWITCH_TOPIC_PROMPT = PromptTemplate(
    input_variables=["app", "instruction", "feature"],
    template="""You are a smart task instruction rewriter for mobile phone tasks. I will provide you with a task instruction for completion and the functionality of an APP on a mobile phone. 
    Rewrite #Given Instruction# by switching the topic for the same APP, keeping the domain and difficulty level similar, and create #New Instruction#.
""" + POST_PROMPT
)


class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)


class WizardLMAgent(BaseModel):
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    num_search_results: int = Field(3, description="Number of pages per Google search")
    text_transformer: BaseDocumentTransformer = Field(Html2TextTransformer(), description="text transformer")
    text_splitter: RecursiveCharacterTextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150),
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )
    app_name: List[str] = Field(
        default_factory=list, description="APP name"
    )
    tasks: List[str] = Field(default_factory=list, description="List of APP tasks.")
    llm: ChatOpenAI = Field(..., description="LLM model")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def web_search(self, questions):
        search = GoogleSerperAPIWrapper(k=self.num_search_results)

        def clean_search_query(query: str) -> str:
            # Some search tools (e.g., Google) will
            # fail to return results if query has a
            # leading digit: 1. "LangCh..."
            # Check if the first character is a digit
            if query[0].isdigit():
                # Find the position of the first quote
                first_quote_pos = query.find('"')
                if first_quote_pos != -1:
                    # Extract the part of the string after the quote
                    query = query[first_quote_pos + 1:]
                    # Remove the trailing quote if present
                    if query.endswith('"'):
                        query = query[:-1]
            return query.strip()

        def search_tool(query: str, num_search_results: int = 1) -> List[dict]:
            query_clean = query
            result = search.results(query_clean)
            return result["organic"]

        # print(f"Questions for Google Search: {questions}")
        # Get urls
        # print("Searching for relevant urls...")
        urls_to_look = []
        for q in questions:
            # Google search
            search_results = search_tool(q, self.num_search_results)
            # print("Searching for relevant urls...")
            # print(f"Search results: {search_results}")
            for res in search_results:
                if res.get("link", None):
                    if ".pdf" in res["link"] or "youtube" in res["link"] or "androidpolice" in res[
                        "link"] or "xda-developers" in res["link"] or "www.makeuseof.com" in res[
                        "link"] or "support.google.com" in res["link"] or "www.howtogeek.com" in res[
                        "link"] or "davinp1.webs.com" in res["link"] or "www.onboard.upenn.edu" in res[
                        "link"] or "medium.com" in res["link"] or "www.pocket-lint.com" in res[
                        "link"] or "www.pulmonaryfibrosis.org" in res["link"]:
                        continue
                    urls_to_look.append(res["link"])
        # Relevant urls
        urls = set(urls_to_look)
        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))
        # print(f"New URLs to load: {new_urls}")
        if new_urls:
            loader = AsyncHtmlLoader(new_urls)
            # print("Indexing new urls...")
            docs = loader.load()
            docs = list(self.text_transformer.transform_documents(docs))
            docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(docs)

    def create_seed_tasks(self, web_search=True):
        if not os.path.exists(f"tasks/{'_'.join(self.app_name)}_seed.txt"):
            prompt = TASK_SEED_PROMPT if len(self.app_name) == 1 else CROSS_TASK_SEED_PROMPT
            seed_task_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                output_parser=CommaSeparatedListOutputParser(),
                output_key="template"
            )
            for app in self.app_name:
                try:
                    docs = WikipediaLoader(query=app, load_max_docs=1).load()
                    docs = list(self.text_transformer.transform_documents(docs))
                    docs = self.text_splitter.split_documents(docs)

                    self.vectorstore.add_documents(docs)
                except Exception as e:
                    print(f"cannot find {app}")
                    print(str(e))

            if web_search:
                if len(self.app_name) == 1:
                    questions = [f"how to use {self.app_name}", f"{self.app_name} usage instructions",
                                 f"{self.app_name} quick start guides", f"{self.app_name} cheat sheets",
                                 f"{self.app_name} productivity guides", f"use {self.app_name} step-by-step",
                                 f"tips and tricks for {self.app_name}", f"{self.app_name} for beginners",
                                 f"{self.app_name} tutorial", f"getting started with {self.app_name}",
                                 f"introduction to {self.app_name}"]
                else:
                    app_name = ["\"" + a + "\"" for a in self.app_name]
                    comb = " and ".join(app_name)
                    questions = [f"{comb} collaboration features", f"How to use {comb} together for tasks",
                                 f"Integration between {comb} for productivity",
                                 f"Collaborative task management with {comb}",
                                 f"{comb} integration for work and productivity", f"Productivity tips with {comb}"]
                self.web_search(questions)

            qa_chain = RetrievalQA.from_chain_type(llm, retriever=self.vectorstore.as_retriever(), verbose=True,
                                                   chain_type="stuff", output_key="feature")
            if len(self.app_name) == 1:
                query = f"what are the features and functions of {self.app_name}?"
            else:
                query = f"what users' tasks can {' and '.join(self.app_name)} complete?"
            features = qa_chain.run(query=query)
            print(features)
            response = seed_task_chain.run(feature=features, app=' and '.join(self.app_name))
            print(response)
            with open(f"tasks/{'_'.join(self.app_name)}_seed.txt", "w") as f:
                f.write(", ".join(response))
        with open(f"tasks/{'_'.join(self.app_name)}_seed.txt", "r") as f:
            self.tasks = [r.strip() for r in f.readlines()]

    def mutate(self, iter_num=10):
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(), verbose=True,
                                               chain_type="stuff", output_key="feature")

        feature = qa_chain.run(query=f"what is the features and functions of {' and '.join(self.app_name)} APP?")
        print(feature)

        for i in range(iter_num):
            print(f"iter {i}...")
            evolve_prompt = np.random.choice(
                [ADD_CONSTRAINTS_PROMPT, COMPLICATE_PROMPT, DEEPEN_PROMPT, SWITCH_TOPIC_PROMPT,
                 INCREASE_REASONING_PROMPT, CONCRETIZE_PROMPT])

            llm_chain = LLMChain(
                llm=self.llm,
                prompt=evolve_prompt
            )

            selected_tasks = np.random.choice(self.tasks, 16)
            response = llm_chain.apply(
                [{"feature": feature, "app": ' and '.join(self.app_name), "instruction": task} for task
                 in selected_tasks])
            new_tasks = []
            for before, after in zip(selected_tasks, response):
                after = after["text"].lower()
                # Elimination Evolving
                if (before == after
                        or after in self.tasks
                        or "n/a" in after
                        or "how can i assist" in after
                        or "as an ai" in after
                        or "ai assistant" in after
                        or "sorry" in after
                        or "new instruction" in after
                        or re.match(r".*<.+>.*", after)):
                    continue
                new_tasks.append(after.strip())
            self.tasks.extend(list(set(new_tasks)))
            with open(f"tasks/{'_'.join(self.app_name)}/iter_{i + 1}.txt", "w") as f:
                f.write("\n".join(self.tasks))

    def self_evolve(self, iter_num=5):
        self.create_seed_tasks()
        self.mutate(iter_num=iter_num)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    all_apps = ["Google Messages", "Google Contacts", "Google Drive", "Slack", "Gmail", "Google Weather", "Google Maps",
                "Chrome", "Android Camera", "Google Clock", "Google Calendar", "YouTube", "Android Setting",
                "Google Photos"]
    llm = AzureChatOpenAI(deployment_name=os.environ["AZURE_ENGINE"],
                          openai_api_key=os.environ["AZURE_OPENAI_KEY"],
                          openai_api_base=os.environ["AZURE_OPENAI_BASE"],
                          openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
                          temperature=0.)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    for app_n in itertools.combinations(all_apps, 1):
        print(app_n)

        vectorstore = Chroma(collection_name='_'.join(app_n),
                             embedding_function=HuggingFaceEmbeddings(),
                             persist_directory=f"./chroma_db_apps")
        agent = WizardLMAgent(app_name=app_n, vectorstore=vectorstore, llm=llm, num_search_results=5,
                              text_splitter=text_splitter,
                              url_database=[])
        try:
            agent.create_seed_tasks()
        except Exception as e:
            print(str(e))
        print(app_n)

