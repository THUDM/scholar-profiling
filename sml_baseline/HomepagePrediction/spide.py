# import gevent
import requests
# import pickle
import os
import re
import time
# import queue
# from lxml import etree
from urllib import parse
# from tqdm import trange
# from requests.compat import urlparse
from pyquery import PyQuery as pq
import sys
if sys.version_info[0] > 2:
    from urllib.parse import urlparse, parse_qs
else:
    from urlparse import urlparse, parse_qs
#
# base_url = 'https://www.google.com/search?q='
# tem = pd.read_excel('tocrawldata(1).xlsx')
# names = tem[2].tolist()
# orgs = tem[3].tolist()
# ids = tem[1]
#
# targets = []
# for i, name in enumerate(names):
#     query = f'"{name}"+{orgs[i]}'
#     targets.append([ids[i], base_url+query])
#
# n = 1
# batch_size = math.ceil(len(targets)/n)
# for i in range(n):
#     start = i * batch_size
#     end = min(start+batch_size, len(targets))
#     with open(f'targets_{str(i)}.pkl', 'wb') as f:
#         pickle.dump(targets[start:end], f)
# sess = requests.session()
# sess.get('https://www.google.com')
# base_url = 'https://www.google.com/search?q='
# data_dir = 'google_search_data'
# headers = {
#     'Accept': 'text/html;charset=utf-8',
# }

def refresh_sess():
    global sess
    sess.close()
    sess = requests.session()
    sess.get('https://www.google.com')
    print(sess.cookies)


def url_extract(url):
    url = parse.unquote(url)
    url = re.findall('https?.*&sa', url)
    return url[0][:-3]


def filter_link(link):
    """
    Returns None if the link doesn't yield a valid result.
    Token from https://github.com/MarioVilas/google
    :return: a valid result
    """
    try:
        # Valid results are absolute URLs not pointing to a Google domain
        # like images.google.com or googleusercontent.com
        o = urlparse(link, 'http')
        if o.netloc:
            return link
        # Decode hidden URLs.
        if link.startswith('/url?'):
            link = parse_qs(o.query)['q'][0]
            # Valid results are absolute URLs not pointing to a Google domain
            # like images.google.com or googleusercontent.com
            o = urlparse(link, 'http')
            if o.netloc:
                return link
        # Otherwise, or on error, return None.
    except Exception as e:
        # LOGGER.exception(e)
        return None


def extra_deep_url(html):
    # root = etree.HTML(html)
    # deep_urls = root.xpath('.//div[@class="ZINbbc xpd O9g5cc uUPGi"]//div[@class="kCrYT"]/a[1]/@href')
    # deep_urls = [url_extract(url) for url in deep_urls]
    # deep_urls = [url for url in deep_urls if url is not None and 'pdf' not in url.lower()]
    pq_content = pq(html)
    res = []
    for p in pq_content.items('a'):
        if p.attr('href').startswith('/url?q='):
            pa = p.parent()
            if pa.is_('div'):
                ppa = pa.parent()
                if ppa.attr('class') is not None:
                    result = {}
                    result['title'] = p('h3').eq(0).text()
                    result['url_path'] = p('div').eq(1).text()
                    href = p.attr('href')
                    if href:
                        url = filter_link(href)
                        result['url'] = url
                    else:
                        result['url'] = ''
                    text = ppa('div').eq(0).text()
                    result['text'] = text
                    res.append(result)
    return res


def crawl_google(query, url):
    query = re.sub('\"', '', query)
    try:
        deep_urls = []
        for x in range(2):
            if x == 1:
                if 'start=10' in resp.text:
                    url += '&start=10'
                else:
                    break
            print(url)
            headers.pop('Host', None)
            resp = sess.get(url, headers=headers)
            # with gevent.Timeout(600*5, ValueError(f'cant reach google {url}')) as timeout:
            while 'captcha' in resp.text:
                print('captcha')
                time.sleep(600)
                refresh_sess()
                resp = sess.get(url, headers=headers)
            deep_urls.extend(extra_deep_url(resp.text))

            # with open(os.path.join(data_dir, f'{str(query)}_s{str(x+1)}.html'), 'w', encoding='utf-8') as f:
            #     f.write(resp.text)
            time.sleep(10)

        if len(deep_urls) < 5:
            raise ValueError(f'content not found enough search result: {url}')

        # for i, res_url in enumerate(deep_urls):
        #     new_id = query + f'_{str(i)}'
        #     deep_queue.put([new_id, res_url['url']])
        return deep_urls

    except Exception as e:
        with open('log_google.txt', 'a', encoding='utf-8') as f:
            f.write(str(query) + str(type(e)) + '\t' + str(e))
            f.write('\n')

    # time.sleep(50)


def crawl_deep(url):
    try:
        # with gevent.Timeout(10, ValueError(f'cant reach or too large {url}')) as timeout:
        headers.pop('Host', None)
        headers['Host'] = urlparse(url).hostname
        resp = requests.get(url, headers=headers, allow_redirects=True)
        # resp.encoding = 'UTF-8'
        if len(resp.text) < 100:
            raise ValueError(f'cant reach ' + str(url))
        if resp.status_code != 200:
            return ''
        return resp.content
    except Exception as e:
        with open('log_deep.txt', 'a', encoding='utf-8') as f:
            f.write(str(url) + str(type(e)) + '\t' + str(e))
            f.write('\n')
        return ''