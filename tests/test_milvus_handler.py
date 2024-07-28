import unittest
from unittest.mock import MagicMock, patch
from pymilvus import CollectionSchema, FieldSchema, DataType
from rag.database.milvus_handler import MilvusHandler


class TestMilvusHandler(unittest.TestCase):
    def setUp(self):
        # Create a MilvusHandler instance for testing
        self.handler = MilvusHandler(host="localhost", port="19530")
        # Define a default schema for the collection
        self.handler.define_default_schema("test_collection")
        # Mocking Collection creation
        self.collection = MagicMock()
        self.collection.name = "test_collection"
        self.collection.has_index.return_value = False
        self.collection.num_entities = 100
        self.collection.insert.return_value = None
        self.collection.search.return_value = []
        patch('pymilvus.Collection', return_value=self.collection).start()
        patch('pymilvus.utility.has_collection', return_value=True).start()

    @patch('pymilvus.connections.connect')
    def test_connect(self, mock_connect):
        self.handler.connect()
        mock_connect.assert_called_with(alias="default", host="localhost", port="19530")

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_create_collection(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = False
        mock_collection.return_value = self.collection
        self.handler.create_collection("test_collection")
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", schema=self.handler.schemas.get("test_collection"), using="default")
        self.handler.create_index.assert_called_with("test_collection")

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_delete_collection(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = True
        mock_collection.return_value = self.collection
        self.handler.delete("test_collection")
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", using="default")
        self.collection.drop.assert_called_once()

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_count_entities(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = True
        mock_collection.return_value = self.collection
        count = self.handler.count("test_collection")
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", using="default")
        self.assertEqual(count, 100)

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_insert(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = True
        mock_collection.return_value = self.collection
        vectors = [[0.1, 0.2, 0.3] * 43]  # Adjust vector dimension to match schema
        self.handler.insert(vectors, "test_collection")
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", using="default")
        self.collection.insert.assert_called_with(vectors)
        self.handler.create_index.assert_called_with("test_collection")

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_search(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = True
        mock_collection.return_value = self.collection
        vectors = [[0.1, 0.2, 0.3] * 43]  # Adjust vector dimension to match schema
        top_k = 5
        results = self.handler.search(vectors, "test_collection", top_k)
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", using="default")
        self.collection.search.assert_called_with(
            vectors, "embedding", self.handler.search_params, top_k
        )

    @patch('pymilvus.utility.has_collection')
    @patch('pymilvus.Collection')
    def test_check_index_status(self, mock_collection, mock_has_collection):
        mock_has_collection.return_value = True
        mock_collection.return_value = self.collection
        self.handler.check_index_status("test_collection")
        mock_has_collection.assert_called_with("test_collection", using="default")
        mock_collection.assert_called_with(name="test_collection", using="default")
        self.collection.has_index.assert_called_once()

    def tearDown(self):
        # Cleanup code if needed
        pass

if __name__ == '__main__':
    unittest.main()
