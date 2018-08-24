from pymongo import MongoClient
from lrf.resources.configs.mongodbconfig import mongodb_config

def setUp():
    mongo_client = MongoClient(mongodb_config['onespace_host'], mongodb_config['port'])
    mongo_db_singhose = mongo_client[mongodb_config['db']]      # DB NAME
    mongo_collection_articles = mongo_db_singhose[mongodb_config['tweet_collection']]     # Collection Name
    mongo_collection_articles.create_index("link",unique=True)
    pass


def getConnection(db, collection):
    mongo_client = MongoClient(mongodb_config['host'], mongodb_config['port'])
    mongo_db_singhose = mongo_client[db]
    mongo_collection = mongo_db_singhose[collection]
    return mongo_collection