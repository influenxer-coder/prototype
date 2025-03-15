from typing import List, Optional

from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5


class VectorDBService:
    def __init__(self, client):
        self.client = client

    def create_collection(self, schema: dict) -> bool:
        collection_name = schema.get('collection_name', None)
        properties = schema.get('properties', None)
        vectorizer = schema.get('vectorizer', None)

        if not collection_name or not properties or not vectorizer:
            print(f"Error: Incorrect schema object - {schema}")
            return False

        try:
            if self.client.collections.exists(collection_name):
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

    def batch_add(self, schema: dict, records: List[dict]):
        collection_name = schema.get('collection_name', None)
        primary_key = schema.get('primary_key', None)

        if not collection_name or not primary_key:
            print(f"Error: Incorrect schema object - {schema}")
            return False

        if not self.client.collections.exists(collection_name):
            print(f"Error: Collection does not exist - {collection_name}")
            return False

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

    def record_exists(self, schema: dict, primary_key: any) -> bool:
        collection_name = schema.get('collection_name', None)

        if not self.client.collections.exists(collection_name):
            print(f"Error: Collection does not exist - {collection_name}")
            return False

        record_uuid = generate_uuid5(primary_key)
        collection = self.client.collections.get(collection_name)

        if collection.data.exists(record_uuid):
            print(f"Record exists in Vector DB already: {record_uuid}")
            return True
        return False

    def search(self, schema: dict, query: str, limit: int = 5, offset: int = 0) -> Optional[List[dict]]:

        collection_name = schema.get('collection_name', None)
        if not self.client.collections.exists(collection_name):
            print(f"Error: Collection does not exist - {collection_name}")
            return None

        collection = self.client.collections.get(collection_name)
        try:
            response = collection.query.hybrid(
                query=query,
                filters=(
                        Filter.by_property("shooting_style").equal("Problem - Solution") &
                        Filter.by_property("impact_score").greater_than(50)
                ),
                limit=limit,
                offset=offset
            )
            return response.objects
        except Exception as e:
            print(f"Error in search query: {e}")
            return None
