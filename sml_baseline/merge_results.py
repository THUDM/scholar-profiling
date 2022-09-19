import os
import json
import pandas as pd
import argparse

import settings

# df_test=pd.read_excel("../data/new_test.xlsx",keep_default_na=False)
df_test = pd.read_excel(os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx"), keep_default_na=False)

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str, default="xgboost", help='classifier')
args = parser.parse_args()

fw=open("output/sml/sml_predict_{}.json".format(args.classifier),'w',encoding="utf-8")

aid_to_gender = {}
with open("output/sml/gender_predict.json") as rf:
    for i, line in enumerate(rf):
        cur_item = json.loads(line)
        aid_to_gender[cur_item["id"]] = cur_item["gender"]

aid_to_hp = {}
with open("output/sml/homepage_predict_{}.json".format(args.classifier)) as rf:
    for i, line in enumerate(rf):
        cur_item = json.loads(line)
        aid_to_hp[cur_item["id"]] = cur_item["homepage"]


aid_to_title = {}
with open("output/sml/test_title_predict1.json") as rf:
    data = json.load(rf)
    for item in data:
        aid_to_title[item["id"]] = item["title"]

for i,row in df_test.iterrows():
    id=row["id"]
    name=row["name"]
    org=row["org"]
    predict_data={"id":id,"name":name,"org":org,"homepage": aid_to_hp[id], "title": aid_to_title[id], "gender": aid_to_gender[id]}
    title_data = json.dumps(predict_data)
    fw.write(title_data + "\n")
fw.close()
