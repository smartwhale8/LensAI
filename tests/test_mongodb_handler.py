import unittest
from pymongo.errors import InvalidOperation
from rag.database.mongodb_handler import MongoDBHandler

class TestMongoDBHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the MongoDB connection
        cls.mongo_handler = MongoDBHandler('localhost', 27017)
        cls.test_collection = "test_collection"
        print(f"Collection name: {cls.test_collection}, Type: {type(cls.test_collection)}")
        cls.test_db = cls.mongo_handler.db

    def setUp(self):
        # Ensure the test collection is empty before each test
        self.mongo_handler.delete_all(self.test_collection)

    def tearDown(self):
        # Clean up the test collection after each test
        self.mongo_handler.delete_all(self.test_collection)

    @classmethod
    def tearDownClass(cls):
        # Close the MongoDB connection
        cls.mongo_handler.close()

    def test_insert_and_find_one(self):
        # Insert a document into the test collection
        sample_data = {'name': 'Alice', 'age': 30, 'occupation': 'Engineer'}
        self.mongo_handler.insert(self.test_collection, sample_data)
        
        # Find the inserted document in the test collection
        query = {'name': 'Alice'}
        result = self.mongo_handler.find_one(self.test_collection, query)
        
        # Check that the result matches the inserted document
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'Alice')
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['occupation'], 'Engineer')
    
    def test_update(self):
        # Insert a document into the test collection
        sample_data = {'name': 'Jane Doe', 'age': 26, 'occupation': 'Designer'}
        self.mongo_handler.insert(self.test_collection, sample_data)
        self.mongo_handler.update(self.test_collection, {'name': 'Jane Doe'}, {'age': 31})
        # Find the updated document in the test collection
        result = self.mongo_handler.find_one(self.test_collection, {'name': 'Jane Doe'})
        self.assertIsNotNone(result)
        self.assertEqual(result['age'], 31)

    def test_delete(self):
        # Insert a document into the test collection
        sample_data = {'name': 'John Smith', 'age': 35, 'occupation': 'Manager'}
        self.mongo_handler.insert(self.test_collection, sample_data)
        # Delete the inserted document from the test collection
        self.mongo_handler.delete(self.test_collection, {'name': 'John Smith'})
        # Find the deleted document in the test collection
        result = self.mongo_handler.find_one(self.test_collection, {'name': 'John Smith'})
        self.assertIsNone(result)
    
    """
    def test_delete_all(self):
        # Insert two documents into the test collection
        sample_data1 = {'name': 'Bob Brown', 'age': 40, 'occupation': 'Developer'}
        self.mongo_handler.insert(self.test_collection, sample_data1)

        sample_data2 = {'name': 'Charlie Green', 'age': 45, 'occupation': 'Architect'}
        self.mongo_handler.insert(self.test_collection, sample_data2)
        # Delete all documents from the test collection
        self.mongo_handler.delete_all(self.test_collection)
        # Find all documents in the test collection
        result = list(self.mongo_handler.find(self.test_collection, {}))
        self.assertEqual(len(result), 0)
    """
    def test_close(self):
        # Close the MongoDB connection
        self.mongo_handler.close()
        with self.assertRaises(InvalidOperation):
            self.mongo_handler.find_one(self.test_collection, {})
        # Reopen the connection for subsequent tests 
        # This ensures that the reopened connection is available for all test methods in the class.
        #self.mongo_handler = MongoDBHandler('localhost', 27017) -- incorrect way to reopen connection

        # The following assigns a new MongoDBHandler instance to the class attribute mongo;
        # each test class and all methods share this same instance
        self.__class__.mongo_handler = MongoDBHandler('localhost', 27017)

if __name__ == '__main__':
    unittest.main()