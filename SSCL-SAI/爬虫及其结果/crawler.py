import time
import requests
import re
import json
import openpyxl


class CommentSpider:
    """
    爬取京东
    """
    commentUrl = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&" \
                 "productId={0}&score=0&sortType=5&page={1}&pageSize=10&isShadowSku=0&rid=0&fold=1"

    origin_reffer = "https://item.jd.com/{0}.html"

    sleep_seconds = 2

    def __init__(self, productId):
        self.productId = productId

    def build_headers(self, next_page_cookie=None):
        Referer = self.origin_reffer.format(self.productId)

        if next_page_cookie is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36",
                "Referer": Referer,
                "Host": "club.jd.com",
                "Cookie": "unpl=V2_ZzNtbRFfQBJzWxQEfB9UAWJQQF9KBBQVdVhOXHpOXwJkUUBfclRCFnUUR1RnGFQUZAEZXkJcRhZFCEdkeBBVAWMDE1VGZxBFLV0CFSNGF1wjU00zQwBBQHcJFF0uSgwDYgcaDhFTQEJ2XBVQL0oMDDdRFAhyZ0AVRQhHZHsYXgdgBhRVSlBzJXI4dmR9HlsCYQEiXHJWc1chVE9UeR1fBioDE19AUEYTfQBBZHopXw%3d%3d; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|0f3d30c8dba7459bb52f2eb5eba8ac7d_0_b9267bca67954bc39bf11a990f262cc3|1610205703260; __jdu=1033647996; areaId=2; PCSYCityID=CN_310000_310100_0; shshshfpb=wVjD8v2Dr7inEEgCOGiQ9kQ%3D%3D; shshshfpa=6792afdc-4156-d1ed-8c5a-86d979144193-1591804178;"
                          " __jda=122270672.1033647996.1610205703.1610205703.1610205703.1;"
                          " __jdc=122270672; shshshfp=4f2edd84f8946f1594a34d185b2d4b3b;"
                          " 3AB9D23F7A4B3C9B=JVSHRSEP2KT6XOTDLFPMA3CYGKN3L5PI427XN6PJDRZ5PBUY6CV3KWZ6Q6YHQJLZI3BKFST2DHV55MHPYODYFB6MTA;"
                          " ipLoc-djd=2-2830-51803-0; jwotest_product=99; shshshsID=690b740513ddb1e914cdc6870e46c538_4_1610206811306;"
                          " __jdb=122270672.4.1033647996|1.1610205703; "
                          "JSESSIONID=AE587338A97897165F8BCB899525EBF4.s1"
            }
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36",
                "Referer": Referer,
                "Host": "club.jd.com",
                "Cookie": "unpl=V2_ZzNtbRFfQBJzWxQEfB9UAWJQQF9KBBQVdVhOXHpOXwJkUUBfclRCFnUUR1RnGFQUZAEZXkJcRhZFCEdkeBBVAWMDE1VGZxBFLV0CFSNGF1wjU00zQwBBQHcJFF0uSgwDYgcaDhFTQEJ2XBVQL0oMDDdRFAhyZ0AVRQhHZHsYXgdgBhRVSlBzJXI4dmR9HlsCYQEiXHJWc1chVE9UeR1fBioDE19AUEYTfQBBZHopXw%3d%3d; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|0f3d30c8dba7459bb52f2eb5eba8ac7d_0_b9267bca67954bc39bf11a990f262cc3|1610205703260; __jdu=1033647996; areaId=2; PCSYCityID=CN_310000_310100_0; shshshfpb=wVjD8v2Dr7inEEgCOGiQ9kQ%3D%3D; shshshfpa=6792afdc-4156-d1ed-8c5a-86d979144193-1591804178;"
                          " __jda=122270672.1033647996.1610205703.1610205703.1610205703.1;"
                          " __jdc=122270672; shshshfp=4f2edd84f8946f1594a34d185b2d4b3b;"
                          " 3AB9D23F7A4B3C9B=JVSHRSEP2KT6XOTDLFPMA3CYGKN3L5PI427XN6PJDRZ5PBUY6CV3KWZ6Q6YHQJLZI3BKFST2DHV55MHPYODYFB6MTA;"
                          " ipLoc-djd=2-2830-51803-0; jwotest_product=99; shshshsID=690b740513ddb1e914cdc6870e46c538_4_1610206811306;"
                          " __jdb=122270672.4.1033647996|1.1610205703; "
                          "JSESSIONID={0}".format(next_page_cookie)
            }
        return headers

    def get_one_page_comment(self, page=0, next_page_cookie=None):
        url = self.commentUrl.format(self.productId, page)
        res = requests.get(url, headers=self.build_headers(next_page_cookie))
        coms_json = self.parse_text_res(res.text)
        next_page_cookie = requests.utils.dict_from_cookiejar(res.cookies)
        return coms_json, next_page_cookie

    def parse_text_res(self, text):
        match_com = re.findall("fetchJSON_comment98(.*)", text)[0][1:].replace(");", "")
        coms_json = json.loads(match_com)
        return coms_json

    def get_all_comments(self):
        coms_json, next_page_cookie = self.get_one_page_comment()
        maxPage = coms_json['maxPage']
        print("最大的页数：" + str(maxPage))
        comments = coms_json['comments']
        rows = []
        print(f"开始爬取商品，{self.productId}")
        print("第{0}页".format(1))
        for comment in comments:
            content = comment['content'].replace("\n", " ")
            creationTime = comment['creationTime']
            usefulVoteCount = comment['usefulVoteCount']
            row = [
                creationTime, usefulVoteCount, content
            ]
            rows.append(row)

        for page in range(1, int(maxPage)):
            print("第{0}页".format(page))
            res, next_page_cookie = self.get_one_page_comment(page=page, next_page_cookie=next_page_cookie)
            time.sleep(self.sleep_seconds)
            comments = res['comments']
            for comment in comments:
                creationTime = comment['creationTime']
                usefulVoteCount = comment['usefulVoteCount']
                content = comment['content'].replace("\n", " ")
                row = [
                    creationTime, usefulVoteCount, content
                ]
                rows.append(row)
        book = openpyxl.Workbook()
        sheet = book.create_sheet(title=self.productId, index=0)
        columns = ['时间', '点赞', '内容']
        sheet.append(columns)
        for row in rows:
            sheet.append(row)
        book.save(f"{self.productId}.xlsx")
        book.close()


if __name__ == "__main__":
    productIds = [
        "100025374022", '100011269283', '100016777664', '10023365907329', '100011386554'
    ]
    for productId in productIds:
        commetSpider = CommentSpider(productId)
        commetSpider.get_all_comments()
