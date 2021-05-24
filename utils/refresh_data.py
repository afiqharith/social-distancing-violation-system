import os, json

class Initilization:
    def __init__(self):
        self.json_path = os.path.join(os.getcwd(), 'output_data.json')
        self.a = self.init_json()
    
    def init_json(self):
        with open(self.json_path, 'r') as fileIn:
            loaded = json.load(fileIn)
            loaded['data'] = list()
            loaded.update(loaded['data'])
        try:
            with open(self.json_path, 'w') as fileOut:
                json.dump(loaded, fileOut, sort_keys=True)
        except IOError as e:
            print(e)

if __name__ == '__main__':
    Initilization()