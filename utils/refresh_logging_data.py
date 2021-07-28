import json
import os

class LoadJSONFromDisk:
    
    def __init__(self) -> None:
        self.json_path = os.path.join(os.getcwd(), 'temp', 'logging.json')
    
    def load_json(self) -> None:
        with open(self.json_path, 'r') as fileIn:
            self.loaded = json.load(fileIn)
    
    def empty_json(self) -> None:
        self.loaded['data'] = list()
        self.loaded.update(self.loaded['data'])
    
    def update_json(self) -> None:
        with open(self.json_path, 'w') as fileOut:
            json.dump(self.loaded, fileOut, sort_keys=True)

    def __str__(self) -> str:
        return f'[INFO] Refreshing logs'

class RefreshProgramLogs(LoadJSONFromDisk):
    def __init__(self) -> None:
        super().__init__()

        self.load_json()
        self.empty_json()
        self.update_json()
        print(self.__str__())

if __name__ == '__main__':
    RefreshProgramLogs()