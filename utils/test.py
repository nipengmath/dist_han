# -*- coding: utf-8 -*-

import json


with open('data.txt', 'w') as outfile:
    data = {"total": 10}
    json.dump(data, outfile)



if __name__ == "__main__":
    print("ok")
