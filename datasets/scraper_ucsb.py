from bs4 import BeautifulSoup
import requests
import time
import random
import re
import os

search_page_urls = []

for id in range(1,670):
    search_page_urls.append(f"https://ebba.english.ucsb.edu/search_combined/?dt=1600-2000&p={id}")

for search_page_url in search_page_urls:
    page_source = requests.get(search_page_url).text
    transcription_urls = re.findall(r"ballad/[0-9]+/xml", page_source)
    sleep_time = random.randint(3,10)
    # print(f"Sleeping {sleep_time} seconds.")
    time.sleep(sleep_time)
    for trans_id, trans_url in enumerate(transcription_urls):
        ballad_id = trans_url.split("/")[1]
        trans_url = "https://ebba.english.ucsb.edu/" + trans_url
        # print(f"Now working on {trans_url}.")
        trans_source = requests.get(trans_url)
        soup = BeautifulSoup(trans_source.content, 'html.parser')
        title = soup.find('div', id='parttitle').getText()
        title = re.sub(r'[^a-zA-Z0-9_ ]', '', title)
        title = title.strip()
        if len(title) > 30:
            title = title[:30]
        poem_div_all = soup.find_all('div', id='lg')
        ballad_text_full = []
        for poem_div in poem_div_all:
            ballad_text = poem_div.getText().split("\n")
            ballad_text = [line +"\n" for line in ballad_text if len(line.strip()) > 0 and len(line.split()) > 2]
            ballad_text_full.extend(ballad_text)
        with open(f"ballads/{ballad_id} - {title}.txt", 'w') as f:
            f.writelines(ballad_text_full)
        # print(f"Finished ballad id {trans_id}: {title}.")
