from pymongo import MongoClient
import json
import requests
import numpy as np

API_ENDPOINT = "http://10.4.21.156"
SECRET_KEY = "BfyzZNmHaLrPyojwEznAI2BJ0frbVn2PwreM8vzTTu7cTgCByj"
MAX_DEG = 11

mongo_client = MongoClient(
    "mongodb+srv://pratyush:genetic@cluster0.sbi3f.mongodb.net/db?retryWrites=true&w=majority"
)
db = mongo_client.get_database("db")
records = db.errors
# collection = db['data']


def urljoin(root, path=""):
    if path:
        root = "/".join([root.rstrip("/"), path.rstrip("/")])
    return root


def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={"id": id, "vector": vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


def get_errors(raw_vector):
    vector = []
    for i in range(len(raw_vector)):
        # v = max(min(raw_vector[i], 10), -10)
        v = raw_vector[i]
        vector.append(v)

    for i in vector:
        assert 0 <= abs(i) <= 10
    assert len(vector) == MAX_DEG
    data = records.find_one({"vector": vector})
    if data is not None:
        print("db")
        return data["errors"]
    # return [1e10,1e10]
    print("server")
    errors = json.loads(send_request(SECRET_KEY, vector, "geterrors"))
    records.insert_one({"errors": errors, "vector": vector})
    return errors


def get_overfit_vector():
    return json.loads(send_request(SECRET_KEY, [0], "getoverfit"))


def get_parents(limit=10):
    ans = records.aggregate(
        [
            # {"$match": {"errors": {"$elemMatch": {"$lte": 1e13}}}},
            {
                "$project": {
                    "errors": 1,
                    # "score": {
                    #     "$add": [
                    #         {
                    #             "$multiply": [{"$arrayElemAt": ["$errors", 0]}, 0.5]
                    #         },  # training
                    #         {
                    #             "$multiply": [{"$arrayElemAt": ["$errors", 1]}, 0.5]
                    #         },  # validation
                    #     ]
                    # },
                    "training": {"$arrayElemAt": ["$errors", 0]},
                    "validation": {"$arrayElemAt": ["$errors", 1]},
                    "vector": 1,
                }
            },
            {"$match": {"training": {"$gte": 4e12}}},
            {"$match": {"training": {"$lte": 6e12}}},
            {"$match": {"validation": {"$gte": 2e12}}},
            {"$match": {"validation": {"$lte": 4e12}}},
            {"$sort": {"score": 1}},
            {"$limit": limit},
        ]
    )

    # return ans
    # for i in ans:
    # print(i)
    parents = []
    data = []
    for p in ans:
        parents.append(p["vector"])
        d = {
            "MSE": p["errors"],
            # "score": p["score"],
            "vector": p["vector"],
        }
        data.append(d)

    return data


# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    print(get_errors(get_overfit_vector()))
