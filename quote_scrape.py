


import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import getpass
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from bs4 import BeautifulSoup
import requests



page_to_scrape = webdriver.Safari()
page_to_scrape.get("http://quotes.toscrape.com")
soup = BeautifulSoup(requests.get("http://quotes.toscrape.com").text, 'html.parser')
page_to_scrape.find_element(By.LINK_TEXT, "Login").click()

time.sleep(3)
username = page_to_scrape.find_element(By.ID, "username")
password = page_to_scrape.find_element(By.ID, "password")
username.send_keys("admin")

my_pass = getpass.getpass()
password.send_keys(my_pass)
page_to_scrape.find_element(By.CSS_SELECTOR, "input.btn-primary").click()

quotes = soup.findAll('span', attrs={'class':'text'})

#FIND ALL THE ITEMS IN THE PAGE WITH A CLASS ATTRIBUTE OF 'AUTHOR'
#AND STORE THE LIST AS A VARIABLE
authors = soup.findAll('small', attrs={"class":"author"})


file = open("scraped_quotes.csv", "w")
writer = csv.writer(file)

writer.writerow(["QUOTES", "AUTHORS"])

while True:
    quotes = soup.findAll('span', attrs={'class':'text'})
    authors = soup.findAll('small', attrs={"class":"author"})
    for quote, author in zip(quotes, authors):
        print(quote.text + " - " + author.text)
        writer.writerow([quote.text, author.text])
    try:
        page_to_scrape.find_element(By.PARTIAL_LINK_TEXT, "Next").click()
    except NoSuchElementException:
        break
file.close()    