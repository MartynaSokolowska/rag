### RAG Project Setup

This project uses **Elasticsearch (ES)** and **Qdrant** for retrieval and storage of embeddings.  
Below are instructions to set up the environment and run the services locally using Docker.

---

### Elasticsearch (ES)

We use a custom **Elasticsearch** image with the **Morfologik plugin (version 8.19.4)**.

Build the Docker image:

```bash
docker build -t es-morfologik:8.19.4 .
```

Run Elasticsearch container:
```bash
docker run --rm -d --name elastic -p 9200:9200 es-morfologik:8.19.4
```

Verify Elasticsearch is running and the plugin is installed:
```bash
curl -s localhost:9200/_cat/plugins?v
```

You should see a list of installed plugins including Morfologik.

### Qdrant

Qdrant is used as a vector database to store embeddings for semantic search.

Run Qdrant container:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Port 6333 will be used to access the Qdrant REST API.

You can also configure persistent storage and other options as needed.

### Python Dependencies

Install Python packages using pip:
```bash
pip install -r requirements.txt
```

### Running the RAG Pipeline

Once ES and Qdrant are running:

Prepare your dataset and embeddings params 
(You can set used embedding model and documents length in the config - if the data is already retrived it will not be reapeted (so if you wanna collect it again you need to delete the data from set path or change the path :)).
 
Start the RAG retrieval pipeline.

Run queries and check retrieval results.

You can set **the Query** in config and also some other things :>

### Notes

Make sure Docker is installed and running.

If you modify the Elasticsearch or Qdrant ports, update the connection settings in your Python code accordingly.

The Morfologik plugin is required for Polish language support in ES.
