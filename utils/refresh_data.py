from io import FileIO
import os, json

def initialize():
    DATA = os.path.join(os.getcwd(), 'output_data.json')

    with open(DATA) as fileIn:
        loaded = json.load(fileIn)

        loaded['data'] = list()
        loaded.update(loaded['data'])

        try:
            with open(DATA, 'w') as fileOut:
                
                json.dump(loaded, fileOut, sort_keys=True)
        except IOError as e:
            print(e)

if __name__ == '__main__':
    initialize()