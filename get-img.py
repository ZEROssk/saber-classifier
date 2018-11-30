from icrawler.builtin import GoogleImageCrawler

w = input('> word:')

fname = input('> save file name:')

print('Start')

crawler = GoogleImageCrawler(storage={"root_dir": "./data/" + fname})
crawler.crawl(keyword=w, max_num=100)

print('END')
print('save file: ',fname)

