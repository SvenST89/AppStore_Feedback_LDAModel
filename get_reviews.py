import pandas as pd
import numpy as np
import json

from app_store_scraper.app_store import AppStore

class GetReviews:
    def __init__(self, appname: str, appid: str, review_count: int):
        self.appname = appname
        self.appid = appid
        self.review_count = review_count
        
    def scrape_and_store_reviews(self):
        _reviews = AppStore(country="de", app_name=self.appname, app_id=self.appid)
        _reviews.review(how_many=self.review_count)
        cbdict={}
        rdict={}
        for i in range(0, _reviews.reviews_count):
            # transform datetime object into datetime string
            date_obj = _reviews.reviews[i]['date']
            date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
            rdict['date']=date_str
            rdict['review']=_reviews.reviews[i]['review']
            rdict['rating']=_reviews.reviews[i]['rating']
            rdict['isEdited']=_reviews.reviews[i]['isEdited']
            rdict['title']=_reviews.reviews[i]['title']
            rdict['userName']=_reviews.reviews[i]['userName']
            if i not in cbdict.keys():
                cbdict[i]=rdict
        
        # Reviews are stored as json-file. Store it locally, too.
        with open(f'data/{self.appname}.json', 'w', encoding="utf-8") as review_file:
            review_file.write(json.dumps(cbdict, ensure_ascii=False))

        # Reviews stored in "_reviews" variable
        # Transform json format to pandas dataframe to store it as csv
        _df = pd.DataFrame(np.array(_reviews.reviews), columns=['review'])
        df = _df.join(pd.DataFrame(_df.pop('review').tolist()))
        print(f"DF looks like:\n{df.head(10)}\nAnd has types: {df.dtypes}")
        for idx, row in df.iterrows():
            #print(f"Type of row['date']: {type(row['date'])}.")
            date = row['date'].strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "-")
            with open(f"data/documents/{idx}_{date}.txt", "w", encoding="utf-8") as rf:
                rf.write(f"{idx}: "+f"{row['title']}"+"\n"+f"{row['date']}"+f"\nRating: {row['rating']}"+"\n"+f"{row['review']}")
        
        # Convert and store the reviews in csv-format
        df.to_csv(f"data/{self.appname}_reviews.csv")
        return df