from flask import Flask, jsonify
import json
from datetime import datetime

app = Flask(__name__)

def to_json(obj):
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    raise TypeError(f'{obj} is not serializable')

class Response_Data:
    def __init__(self, data=None, status=None, message=None):
        if data is not None and isinstance(data, list):
            self.data = [data1.to_json() for data1 in data]
        else:
            self.data = data
        self.status = status
        self.message = message
    def setData(self, data):
        self.data = data
    
    def to_json(self):
        return {
            "data": self.data,
            "status": self.status,
            "message": self.message
        }


class Organ:
    def __init__(self, name):
        self.name = name
        
    def to_json(self):
        return {
            "name": self.name,
        }


class User:
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age
        self.organ = Organ("TJ")
    
    def to_json(self):
        return {
            "name": self.name,
            "age": self.age,
            "organ": self.organ.to_json(),
        }

users = [
    User("John", 20),
    User("Mary", 21),
    User("Peter", 22)
]

user = users[-1]

@app.route("/datas")
def hello():
    response_data = Response_Data(data=users, status=200, message="datas success")
    json_data = json.dumps(response_data, ensure_ascii=False, default=to_json)
    return json_data

@app.route("/data")
def hello1():
    response_data = Response_Data(data=user, status=200, message="data success")
    json_data = json.dumps(response_data, ensure_ascii=False, default=to_json)
    return json_data

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
