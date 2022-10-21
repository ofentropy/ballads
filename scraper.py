from bs4 import BeautifulSoup
import requests
import html
import re


file = open("/Users/zuyizhao/Documents/GitHub/ballads/UrlsToPoetryFoundationBallads")
urls = file.readlines()

folder = "BalladsPoetryFoundation/"
titles = open("TitlesAuthorsBalladsPoetryFoundation.txt", "w")

for url in urls:
    f = requests.get(url)
    soup = BeautifulSoup(f.text, 'html.parser')
    raw_title_find = soup.find("meta", attrs={'name':'dcterms.Title'})
    if raw_title_find:
        raw_title = "By".join(raw_title_find["content"].split("by"))
        print(raw_title, file=titles)
        raw_title = re.sub(r'\W+', '',raw_title)
        title = "".join(raw_title.split())
        title = html.unescape(title) + ".txt"
        new_file = open(folder+title, 'w')

        poem = soup.find('div',{'class':'o-poem'})
        if poem:
            poemlines = poem.findAll('div')
            for line in poemlines:
                text = html.unescape(line.text.strip())
                print(text, file=new_file)