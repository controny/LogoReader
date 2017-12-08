# -*- coding:utf-8 -*-

import urllib
import urllib2
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool, Value
import os
import logging

DEBUG = False
NUM_PROCESSES = 4

LOGOS_DIR = '../../data/logos'
CACHE_PATH = '../../data/temp'
TO_SEARCH = [
    '雪佛兰',
    '标致',
    '大众',
    '现代',
    '别克',
    '比亚迪',
    '本田',
    '起亚',
    '雪铁龙',
    '丰田',
    '斯柯达',
    '马自达',
    '凯迪拉克',
    '福特',
    '奥迪',
    '日产',
    '三菱',
    '捷豹',
    '斯巴鲁',
    '雷克萨斯'
    ]

class ImagesCrawler(object):
    def __init__(self, logos_dir):
        self.logos_dir = logos_dir
        if DEBUG:
            self.driver = webdriver.Chrome()
        else:
            self.driver = webdriver.PhantomJS()

    def close(self):
        self.driver.close()

    def search(self, word):
        self.word = word
        self.logos_subdir = os.path.join(self.logos_dir, self.word)
        if not os.path.exists(self.logos_subdir):
            os.makedirs(self.logos_subdir)

        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        self.url_file = os.path.join(CACHE_PATH, self.word + '.txt')
        if not os.path.exists(self.url_file):
            self._get_seed_image_urls()
            self._search_by_images()

        self.num_download = 0
        self._download_images()

    def _get_seed_image_urls(self):
        print 'Begin to get seed image urls for "%s"' % self.word
        url_base = "https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&pn=0&gsm=50&ct=&ic=0&lm=-1&width=0&height=0&"
        self.seed_image_urls = []
        self._connect_website(
            url_base + urllib.urlencode({'word': self.word + ' 汽车'}),
            'Fail to search seed images for "%s"' % self.word)

        max_page_num = 2
        page_count = 1
        for i in range(max_page_num):
            try:
                elements = [x.find_element_by_tag_name('img') for x in self.driver.find_elements_by_class_name('imglink')]
                for element in elements:
                    url = element.get_attribute('src')
                    if not url:
                        continue
                    self.seed_image_urls.append(url)
            except Exception, e:
                print 'Fail to collect an image url of "%s" due to %s' % (self.word, e)
            finally:
                # 若每次都模拟点击“下一页”，页面只会在某几个页面中循环，原因不详
                # 因此只能每次模拟点击具体的页码
                elements = self.driver.find_elements_by_class_name('pc')
                for element in elements:
                    if element.text == str(page_count + 1):
                        element.click()
                        break
                page_count += 1

        print 'Totally %d seed images collected for "%s"' % (len(self.seed_image_urls), self.word)
        
    def _search_by_images(self):
        print 'Begin to collect image urls for "%s"' % self.word
        url_base = 'http://image.baidu.com/pcdutu/p_list'
        param = {
            'word': self.word, 
            'queryImageUrl': '', 
            'simid': ['0,0'], 
            'pos': ['moresimi'], 
            'fm': ['result'], 
            'rn': ['30'], 
            'querytype': ['0'], 
            'querySign': ['2270127671,4291584314']
        }

        with open(self.url_file, 'w') as f:
            self.image_urls = []
            for seed_image_url in self.seed_image_urls:
                param['queryImageUrl'] = seed_image_url
                final_url = url_base + '?' + urllib.urlencode(param) + '#activeTab=3'
                self._connect_website(final_url, 'Fail to search "%s" by images' % self.word)

                # 多次滚到页面最底端以加载更多图片
                scroll_times = 5
                scroll_pause = 0.5
                for i in range(scroll_times):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause)

                urls_to_save = self._collect_image_urls()
                f.write('Referer ' + self.driver.current_url + '\n')
                for url in urls_to_save:
                    f.write(url + '\n')
            
        print 'Totally %d image urls collected for "%s"' % (len(self.image_urls), self.word)
            
    def _collect_image_urls(self):
        urls_to_save = []
        elements = [x.find_element_by_tag_name('img') for x in self.driver.find_elements_by_class_name('imglist-item')]
        for element in elements:
            url = element.get_attribute('src')
            if not url:
                continue
            self.image_urls.append(url)
            urls_to_save.append(url)
            if len(self.image_urls) % 50 == 0:
                print '%d urls collected for "%s"' % (len(self.image_urls), self.word)

        return urls_to_save

    def _download_images(self):
        print 'Begin to download images for "%s"' % self.word
        with open(self.url_file, 'r') as f:
            referer = ''
            for line in f.readlines():
                if line.split(' ')[0] == 'Referer':
                    referer = line.split(':')[1].strip()
                else:
                    self._download_single_image(line.strip(), referer)
        print 'Totally %d images downloaded for "%s"' % (self.num_download, self.word)

    def _download_single_image(self, image_url, referer):
        file_name = os.path.join(self.logos_subdir.decode('utf-8'), image_url.replace('/', ''))
        extension = os.path.splitext(file_name)[1]
        if extension == '.gif' or os.path.exists(file_name):
            return
        try:
            # 为躲过百度识图的反爬虫机制，需设置Headers
            headers = {
                'User-Agent': 'Chrome/60.0.3112.113',
                'Referer': referer
            }
            request = urllib2.Request(image_url, headers=headers)
            data = urllib2.urlopen(request, timeout=3).read()
            with open(file_name, 'wb') as f:
                f.write(data)
            self.num_download += 1
        except Exception, e:
            print 'Fail to download an image of "%s" due to %s' % (self.word, e)

        if self.num_download % 50 == 0:
            print '%d images of "%s" downloaded' % (self.num_download, self.word)

    def _connect_website(self, url, message):
        connect = False
        while not connect:
            try:
                self.driver.set_page_load_timeout(5)
                self.driver.get(url)
                connect = True
            except Exception, e:
                print message
                print 'Due to exception:\n%s\nTrying again...' % e


def search(word):
    # 保证最后一定关闭浏览器
    try:
        crawler = ImagesCrawler(LOGOS_DIR)
        crawler.search(word)
    except Exception, e:
        logging.exception(e)
    finally:
        crawler.close()

if __name__ == '__main__':
    # 发生错误自动重启
    flag = True
    while flag:
        try:
            pool = Pool(processes=NUM_PROCESSES)
            pool.map(search, TO_SEARCH)
            flag = False
        except Exception, e:
            logging.exception(e)
