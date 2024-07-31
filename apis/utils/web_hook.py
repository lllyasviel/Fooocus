"""
web hook, send results to given url, or some other place.
"""
import httpx


async def send_result_to_web_hook(url: str, result):
    """
    send result to web hook
    :param url: web hook url
    :param result: result dict
    :return:
    """
    if url is None or url == '':
        return
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0'
    }
    try:
        httpx.post(url, headers=headers, json=result, timeout=5, follow_redirects=True)
    except Exception as e:
        print(e)
    return
