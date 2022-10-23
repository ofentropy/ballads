from fileinput import filename
from bs4 import BeautifulSoup
import requests
import html
import re

file = open("UrlsToBOEBBallads")
urls = file.readlines()

folder = "BalladsBOEBSacredTexts/"
titles = open("TitlesBalladsBOEBSacredTexts.txt", "w")

for url in urls:
    f = requests.get(url)
    soup = BeautifulSoup(f.text, 'html.parser')
    raw_title_find = soup.find("title")
    if raw_title_find:
        raw_title = raw_title_find.text.split(":")[1] + "_" + raw_title_find.text.split(":")[0]
        title = raw_title.split("_")[0]
        print(title, file=titles)
        raw_title = re.sub(r'\W+', '',raw_title)
        file_name = "".join(raw_title.split()) + ".txt"
        new_file = open(folder+file_name, 'w')
        poem = soup.find("font", attrs={"face":"Times Roman,Times New Roman"})
        if poem:
            lines = poem.findAll("p")
            for line in lines:
                if not line.find("a"):
                    linetext = "\n".join(l.strip() for l in html.unescape(line.text.split("\n")))
                    print(linetext, file=new_file)