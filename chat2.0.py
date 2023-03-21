# import logging
# 
# logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)
import os
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import launch_es
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from flask import Flask, request, jsonify, make_response, render_template
from haystack.nodes import OpenAIAnswerGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import Seq2SeqGenerator


from flask_cors import CORS





# Get the host where Elasticsearch is running, default to localhost


doc_dir = r"zui"
all_docs = convert_files_to_docs(dir_path=doc_dir,encoding="utf-8")
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)
docs = preprocessor.process(all_docs)

print(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(docs)}")
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index="document"
)
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)
# Important:
# Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
document_store.update_embeddings(retriever)
# Write documents to document store
document_store.write_documents(docs)

# Add documents embeddings to index
document_store.update_embeddings(retriever=retriever)

generator = OpenAIAnswerGenerator(
    api_key="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",
    model="text-davinci-003",
    max_tokens=50,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    top_k=3,
    temperature=0.9
)
pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
# pipe = ExtractiveQAPipeline(reader, retriever)






app = Flask(__name__)

cors = CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat', methods=['GET'])
def index():
    # handle GET request
    query = request.args.get('msg')
    # print(query['query'])
    prediction = pipe.run(
        query=query,
        params={
            "Retriever": {"top_k": 10},
            # "Reader": {"top_k": 5}
        }
    )
    temp = print_answers(prediction,details="minimum")
    # for tmp in temp:
        # for key, value in tmp.items():
            # print(f"The value of {key} in dictionary {tmp} is of type {type(value)}")
            # value = value.encode("utf-8")
            # print(f"The value of {key} in dictionary {tmp} is of type {type(value)}")
    # print(prediction.encode('utf-8'))
    res = make_response(jsonify(temp[0]))
    return res  ## Choose from `minimum`, `medium`, and `all`


if __name__ == '__main__':
    app.run()