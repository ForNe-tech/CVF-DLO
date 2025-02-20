import json
import os

class Read_And_Save_Json():

    def __init__(self):
        super.__init__()

    def read_masks_json(self, path=None):
        file = open(path, 'r', encoding='utf-8')
        content = file.read()
        if content.startswith(u'\ufeff'):
            content = content.encode('utf8')[3:].decode('utf8')
        Dict_mask = json.loads(content)
        # print(Dict_mask)
        file.close()
        # os.remove(path)
        return Dict_mask

    def read_masks_json_noremove(self, path=None):
        file = open(path, 'r', encoding='utf-8')
        content = file.read()
        Dict_mask = json.loads(content)
        # print(Dict_mask)
        file.close()
        return Dict_mask

    def save_results_json(self, Dict=None, path=None):
        Json_result = json.dumps(Dict, ensure_ascii=False)
        file = open(path, 'w', encoding='utf-8')
        file.write(Json_result)
        file.close()

if __name__ == "__main__":
    Read_And_Save_Json.read_masks_json(self=None)