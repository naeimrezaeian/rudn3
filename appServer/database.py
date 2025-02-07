from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
import uuid
import json
from bson import ObjectId

class Database:
    def __init__(self,db_name: str):
        self.db_name=db_name
        self.collection1_name = "records"        
        self.client=None
        self.db=None
        

        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.client.admin.command('ping')  
            #self.client.drop_database(self.db_name)          
        except ConnectionFailure:
                raise Exception("Failed to connect to MongoDB server")
        
        self.db_list = self.client.list_database_names()
        if self.db_name in self.db_list:
            self.db=self.client[db_name]
            print(f"Database '{self.db_name}' is connected.")
        else:            
            print(f"Database '{self.db_name}' not found.")
            self.createDb()
            print(f"Database '{self.db_name}' is created.")
    def createDb(self):
        self.db = self.client[self.db_name]
        
        collection1_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["title","desc","filename","status","created_at","results"],
                "properties": {
                    "_id": {"bsonType": "objectId", "description": "must be an objectId"},
                    "title": {"bsonType": "string", "description": "must be a string and is required"},
                    "desc": {"bsonType": "string", "description": "must be a string and is required"},
                    "filename": {"bsonType": "string", "description": "must be a string and is required"},
                    "status": {"bsonType": "int", "description": "must be an integer and is required"},
                    "created_at": {"bsonType": "date", "description": "must be a date and is required"},
                    "results": {"bsonType": "string", "description": "must be a string"},
                }
            }
        }

        
        self.db.create_collection(self.collection1_name, validator=collection1_schema)
       

    def addRecord(self,title: str, desc: str,filename:str ):
         

         collection = self.db["records"]
         
         
         record_data={
              "title":title,
              "desc":desc,
              "filename":filename,
              "status":3,
              "created_at": datetime.now(),
              "results":"" 
         }
         result= collection.insert_one(record_data)
         print(f"User added with ID: {result.inserted_id}")
    def addMetric(self,record_id: int, data: str,method_id: int,status: int ):
         collection = self.db["metrics"]
         metric_data={
              "record_id":record_id,
              "data":data,
              "method_id":method_id,
              "status":status
         }
         result= collection.insert_one(metric_data)
         print(f"User added with ID: {result.inserted_id}")
    def getRecords(self):
         
         collection = self.db["records"]
         cursor = collection.find({"status": {"$gt": 0}}).sort("created_at", -1)
         
         data_list = []
         for document in cursor: 
               
            document["_id"] = str(document["_id"])
            timestamp=document["created_at"]
            document["date"] = timestamp.date()
            document["time"] = timestamp.strftime("%H:%M:%S")
            del document["created_at"]
            del document["results"]
            data_list.append(document)
         json_data = json.dumps(data_list, default=str)
         return json_data
    def deleteRecords(self,id):
        collection = self.db["records"]
        result = collection.update_one(
        {"_id": id},
        {"$set": {"status": 0}})
        return result
    async def searchRecords(self,q):
        collection = self.db["records"]
        cursor = collection.find({"title": {"$regex": q, "$options": "i"},"status": { "$gt": 0 }}).sort("created_at", -1)
        
        data_list = []
        for document in cursor: 
                       
            document["_id"] = str(document["_id"])
            timestamp=document["created_at"]
            document["date"] = timestamp.date()
            document["time"] = timestamp.strftime("%H:%M:%S")
            del document["created_at"]
            data_list.append(document)
        json_data = json.dumps(data_list, default=str)
        return json_data
    def get_tasks(self):
        collection = self.db["records"]
        record = collection.find_one_and_update(
            {"status": 3},
            {"$set": {"status": 2}},  
            sort=[("_id", 1)]  
        )
        return record
    def update_task(self,id,results,status):
        collection = self.db["records"]
        collection.update_one({"_id": id}, {"$set": {"status": status,"results":results}})
    def getRecordById(self,id):
            collection = self.db["records"]
            document = collection.find_one({'_id': ObjectId(id)})
            json_data = json.dumps(document, ensure_ascii=False,default=str)
            return json_data


         

         
        
        
            
         