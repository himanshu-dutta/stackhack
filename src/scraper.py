import os
import time
import json
import dotenv
import argparse
import requests
from pprint import pprint

dotenv.load_dotenv()

REQUEST_LIMIT = 30
NUM_REQUESTS = 0
SLEEP_TIME = 60

BASE_URL = "https://api.stackexchange.com/2.3/questions?page={page_no}&pagesize=100&order=desc&sort=votes&site=cs&filter=-pStWFGpkL.xmiz5.PpT-8zKfMTb6lRw6l1_YEn(V43a*" + f"&key={os.environ['SO_KEY']}&access_token={os.environ['SO_ACCESS_TOKEN']}"


def sanity_check():
    global NUM_REQUESTS
    global REQUEST_LIMIT

    NUM_REQUESTS += 1
    if NUM_REQUESTS >= REQUEST_LIMIT:
        print(f"Reached request limit. Sleeping for {SLEEP_TIME} seconds.")
        time.sleep(SLEEP_TIME)
        NUM_REQUESTS = 0


def main(save_path: str, save_after_every_fetch: bool = True):
    questions = list()
    idx=0
    has_more = True
    quota_remaining = True

    while has_more and quota_remaining:
        idx += 1
        url = BASE_URL.format(page_no=idx)
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Error while making request: {resp.status_code}")
            pprint(resp.json())
        
        data: dict = resp.json()
        has_more = data.get("has_more", False)
        quota_remaining = True if data.get("quota_remaining", 1) else False
        questions.extend(data.get("items", []))

        if save_after_every_fetch:
            with open(save_path, "w") as fp:
                json.dump(questions, fp, indent=None)
        
        print(f"Quota Remaining: ", data.get("quota_remaining", -1))
        sanity_check()

        backoff_secs = data.get("backoff", 0)
        if backoff_secs:
            time.sleep(backoff_secs)
            continue
        time.sleep(34 / 1000) # not more than 30 requests per second

    with open(save_path, "w") as fp:
        json.dump(questions, fp, indent=None)

    print(f"Number of data points collected: {len(questions)}")
    print(f'Read data till page: {idx}, Quota Remaining: {quota_remaining}, Has More: {has_more}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("StackHack - Scraper")
    parser.add_argument("-s", "--save_path", type=str, required=True, help="where to store the downloaded data as json")
    args = parser.parse_args()

    main(save_path = args.save_path)