import grequests as g
import requests as req
import json
from datetime import datetime, timedelta


api = 'https://api.flickr.com/services/rest'
base_params = {'format': 'json', 'api_key': ''}
DEBUG = True


def flickr(**kwargs):
    '''
    Send request to flickr API
    -> json
    '''
    params = base_params
    for k, v in kwargs.items():
        params[k] = v
    res = req.get(api, params=params)
    kwargs.pop('method')
    print(kwargs)
    return json.loads(res.text[14:-1])


def gen_steps():
    res = 0.02
    while res < 32.3:
        yield res
        res *= 2
    yield 32.3


def get_requests(photos: list):
    '''
    Get async requests iterable to download images asynchronously
    -> requests, urls
    '''
    def fallback_url(ph):
        for url in ['url_o', 'url_h', 'url_b']:
            if url in ph.keys():
                return ph[url]

    urls = [u for u in map(fallback_url, photos) if u is not None]

    rs = [g.get(url) for url in urls]
    return rs, urls


def get_1k_flickr(lat: float, lon: float, ts: datetime = None, delta: timedelta = None, count=250):
    '''
    Get around `count` photos near `lat,lon` within `ts` +- `delta` (!) time (not date)
    '''
    if ts is None:
        ts = datetime(1, 1, 1)
    if delta is None:
        delta = timedelta(hours=24)
    norm_ts = ts.replace(year=1, month=1, day=1)
    assert count <= 500

    default_opts = dict(
        method='flickr.photos.search',
        lat=lat, lon=lon,
        extras='url_o,url_h,url_b,tags,machine_tags,date_taken',
        accuracy=16,
    )

    get_photos = lambda **kwargs: flickr(**default_opts, **kwargs)['photos']

    photos = []

    for r in list(gen_steps())[3:]:
        photos = get_photos(page=1, radius=r, per_page=1)
        total_rad = r
        if (int(photos['total']) > 2 * count):
            break

    pics = []
    page = 1
    while page <= photos['pages'] and len(pics) < count:
        batch = get_photos(page=page, radius=total_rad, per_page=count * 2)

        if DEBUG:
            print('pics per page: {}, batch size: {}, pics size: {}'
                  .format(batch['perpage'], len(batch['photo']), len(pics)), end=' ')
            unique_pics = set()
            for p in batch['photo']:
                unique_pics.add(p['id'])
            for p in pics:
                unique_pics.add(p['id'])
            print('unique pics: {}'.format(len(unique_pics)))

        for p in batch['photo']:
            if int(p['datetakenunknown']) == 0:
                t = datetime.strptime(p['datetaken'], '%Y-%m-%d %H:%M:%S')
                t = t.replace(year=1, month=1, day=1)
                if abs(norm_ts - t) <= delta:
                    pics.append(p)
            else:
                pics.append(p)

        page += 1

    return pics


def download_photos(rs, urls):
    '''
    Download photos using requests from `get_requests`
    rs: iterable
    urls: list
    '''
    formats = [u.split('.')[-1] for u in urls]

    import time
    start = time.time()
    photos_content = g.map(rs)
    print('Finished in {}s'.format(time.time() - start))
    return photos_content, formats


def get_photos_for(photos, dirname='pics', file_prefix='pic'):
    rs, urls = get_requests(photos)
    photos_content, formats = download_photos(rs, urls)

    for i in range(len(photos_content)):
        if photos_content[i] is not None:
            filename = '{}/{}{}.{}'.format(dirname, file_prefix, i, formats[i])
            with open(filename, 'wb') as fd:
                fd.write(photos_content[i].content)
