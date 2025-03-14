from weaviate.collections.classes.config import Configure, Property, DataType


def get_schema() -> dict:
    return {
        'collection_name': 'Post',
        'vectorizer': Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-large"
        ),
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
        ],
        'primary_key': 'post_id'
    }
