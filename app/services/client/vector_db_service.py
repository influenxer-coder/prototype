from typing import List

from weaviate.collections.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5


class VectorDBService:
    def __init__(self, client):
        self.client = client
        self.schema = {
            'collection_name': 'Post',
            'vectorizer': Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-large"
            ),
            'primary_key': 'post_id',
            'properties': [
                Property(
                    name="post_id",
                    data_type=DataType.TEXT,
                    skip_vectorization=True
                ),
                Property(
                    name="url",
                    data_type=DataType.TEXT,
                    skip_vectorization=True
                ),
                Property(
                    name="description",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="impact_score",
                    data_type=DataType.NUMBER,
                    skip_vectorization=True
                ),
                Property(
                    name="search_term",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="transcript",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="text_elements",
                    data_type=DataType.TEXT
                ),
                Property(
                    name="shooting_style",
                    data_type=DataType.TEXT,
                    skip_vectorization=True
                ),
                Property(
                    name="object",
                    data_type=DataType.TEXT,
                    skip_vectorization=True,
                    index_filterable=False,
                    index_searchable=False
                )
            ]
        }

    def create_collection(self) -> bool:
        collection_name = self.schema['collection_name']
        properties = self.schema['properties']
        vectorizer = self.schema['vectorizer']

        try:
            if self.client.collections.exists(collection_name):
                print(f"Collection already exists: {collection_name}")
                return True

            self.client.collections.create(
                name=collection_name,
                vectorizer_config=vectorizer,
                properties=properties
            )
            print(f"Collection created: {collection_name}")
            return True
        except Exception as e:
            print(f"Error in creating/checking collection: {e}")
            return False

    def batch_add(self, records: List[dict]):
        collection_name = self.schema['collection_name']
        primary_key = self.schema['primary_key']

        collection = self.client.collections.get(collection_name)

        with collection.batch.dynamic() as batch:
            for record in records:
                record_uuid = generate_uuid5(record[primary_key])
                batch.add_object(
                    properties=record,
                    uuid=record_uuid
                )
                if batch.number_errors > 10:
                    print("Batch import stopped due to excessive errors.")
                    break

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports: {len(failed_objects)}")
            return False

        print(f"Added batch of size {len(records)} to the Vector DB")
        return True

    def record_exists(self, primary_key: any) -> bool:
        collection_name = self.schema['collection_name']

        record_uuid = generate_uuid5(primary_key)
        collection = self.client.collections.get(collection_name)

        if collection.data.exists(record_uuid):
            print(f"Record exists in Vector DB already: {record_uuid}")
            return True
        return False
