from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

urls = [
    "https://nextjs.org/docs/app/building-your-application/routing/parallel-routes",
]

loader = UnstructuredURLLoader(urls=urls)
print(loader.load())

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=0,
    length_function=len,
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

query = "Parallel Routesについて300文字以内で分かりやすく教えて"
answer = index.query(query)
print(answer)
