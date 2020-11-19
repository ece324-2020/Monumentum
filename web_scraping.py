#NOTE:

'''
Lot of the code for functions below is found on the website https://medium.com/@wwwanandsuresh/web-scraping-images-from-google-9084545808a2
We do not represent the first two functions as our own.
'''
from find_ids import return_wikimedia_list
import os
import selenium
from selenium import webdriver
import time
import requests
import os
from PIL import Image
import io
import hashlib
# This is the path I use
#DRIVER_PATH = '/Users/anand/Desktop/chromedriver'
# Put the path for your ChromeDriver here
DRIVER_PATH = '/Users/Admin/Documents/GitHub/Monumentum/chromedriver'


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        #print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                #print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            #print("Found:", len(image_urls), "image links, looking for more ...")
            return image_urls
            time.sleep(30)
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,file_name:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        folder_path = os.path.join(folder_path,file_name)
        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        #print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def id_extractor(extracted_wikimedia_list,image_number=50,images_path='images',DRIVER_PATH = '/Users/Admin/Documents/GitHub/Monumentum/chromedriver'):
    def string_returner(query):
        final_query_string,split_query='',query.split(' ')
        for index,i in enumerate(split_query):
            if i==',' or i=='(' or i==')':
                print('True')
                continue
            final_query_string+=i
            if index != len(split_query)-1:
                final_query_string+='_'
        final_final_string=''
        for s in final_query_string:
            if s==',' or s=='(' or s==')':
                continue
            final_final_string+=s
        final_query_string=final_final_string
        return final_query_string
    wd = webdriver.Chrome(executable_path=DRIVER_PATH)
    queries = extracted_wikimedia_list  #change your set of querries here
    underrepresented = []
    for query in queries:
        wd.get('https://google.com')
        search_box = wd.find_element_by_css_selector('input.gLFyf')
        search_box.send_keys(query[1])
        links = fetch_image_urls(query[1],image_number,wd)
        images_path = images_path
        final_query_string = string_returner(query[1])
        if len(links)<image_number:
            underrepresented += [(final_query_string,len(links))]
            print(final_query_string,len(links))
        for i in links:
            persist_image(images_path,str(query[0]),i)
    wd.quit()
    return underrepresented

if __name__=='__main__':
    extracted_wikimedia_list, images_path = return_wikimedia_list(), 'images'
    try:
        os.mkdir(images_path)
    except:
        pass
    underrepresented=id_extractor(extracted_wikimedia_list)
    print(underrepresented)
