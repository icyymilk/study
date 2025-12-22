import os
import random
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from baiduspider import BaiduSpider
from urllib.parse import urlparse
from urllib import robotparser

# global caches/utilities
ROBOTS_CACHE = {}

class HostRateLimiter:
    def __init__(self, min_delay=3.0, jitter=(0.5, 2.0)):
        self.min_delay = float(min_delay)
        self.jitter = jitter
        self._last = {}

    def sleep(self, host: str):
        now = time.time()
        last = self._last.get(host)
        if last is not None:
            elapsed = now - last
            wait = max(0.0, self.min_delay - elapsed) + random.uniform(*self.jitter)
            if wait > 0:
                time.sleep(wait)
        self._last[host] = time.time()

RATE_LIMITER = HostRateLimiter(min_delay=3.0)

class ResponseCache:
    def __init__(self):
        self._etag = {}
        self._last_modified = {}

    def conditional_headers(self, url: str):
        h = {}
        etag = self._etag.get(url)
        lm = self._last_modified.get(url)
        if etag:
            h["If-None-Match"] = etag
        if lm:
            h["If-Modified-Since"] = lm
        return h

    def update(self, url: str, resp: requests.Response):
        etag = resp.headers.get("ETag")
        if etag:
            self._etag[url] = etag
        lm = resp.headers.get("Last-Modified")
        if lm:
            self._last_modified[url] = lm

RESPONSE_CACHE = ResponseCache()

def build_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]

def build_headers():
    ua = random.choice(UA_POOL)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def build_proxies():
    user = os.getenv("HORO_USERID")
    pwd = os.getenv("HORO_PASSWORD")
    host = os.getenv("HORO_HOST", "dyn.horocn.com")
    port = os.getenv("HORO_PORT", "50000")
    if user and pwd:
        meta = f"http://{user}:{pwd}@{host}:{port}"
        return {"http": meta, "https": meta}
    return None

def polite_get(session, url, headers, proxies=None, timeout=10, referer=None):
    ua = headers.get("User-Agent", UA_POOL[0])
    host = urlparse(url).netloc

    # robots.txt respect
    rp = ROBOTS_CACHE.get(host)
    if rp is None or (time.time() - rp._mtime > 60 * 30):  # refresh every 30 min
        rp = robotparser.RobotFileParser()
        rp.set_url(f"https://{host}/robots.txt")
        try:
            rp.read()
            rp._mtime = time.time()
        except Exception:
            # If robots fetch fails, be conservative with delays
            rp._mtime = time.time()
        ROBOTS_CACHE[host] = rp

    allowed = True
    try:
        allowed = rp.can_fetch(ua, url)
    except Exception:
        allowed = True
    if not allowed:
        raise PermissionError(f"robots.txt disallows fetching: {url}")

    # Determine crawl delay
    delay = None
    try:
        delay = rp.crawl_delay(ua)
    except Exception:
        delay = None
    base_delay = delay if isinstance(delay, (int, float)) else random.uniform(2.0, 5.0)
    time.sleep(base_delay + random.uniform(0.5, 2.0))
    
    # host-level rate limiting
    RATE_LIMITER.sleep(host)
    # combine headers with conditional and referer
    req_headers = dict(headers)
    if referer:
        req_headers["Referer"] = referer
    req_headers.update(RESPONSE_CACHE.conditional_headers(url))

    resp = session.get(url, headers=req_headers, proxies=proxies, timeout=timeout)
    RESPONSE_CACHE.update(url, resp)

    # Gentle handling of rate limiting or anti-bot pages
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        try:
            wait_s = float(retry_after)
        except Exception:
            wait_s = random.uniform(30, 90)
        time.sleep(wait_s)
    elif resp.status_code in (403, 503):
        time.sleep(random.uniform(20, 60))

    text_lower = resp.text[:2000].lower()
    if any(k in text_lower for k in ("captcha", "verify", "滑动验证", "人机验证")):
        # Back off hard and let caller decide next steps
        time.sleep(random.uniform(60, 180))
    return resp

def main():
    session = build_session()
    headers = build_headers()
    proxies = build_proxies()

    try:
        resp = polite_get(session, "https://baike.baidu.com", headers=headers, proxies=proxies, timeout=10)
        print("请求成功，状态码：", resp.status_code)
        print("页面内容片段：", resp.text[:100])
    except Exception as e:
        print("请求失败：", e)

    cookie = os.getenv("BAIDU_COOKIE")
    spider = BaiduSpider(cookie)
    try:
        result = spider.search_baike("王者荣耀", proxies=proxies)
        from pprint import pprint
        pprint(result.plain)
    except Exception as e:
        print("搜索失败：", e)

if __name__ == "__main__":
    main()
