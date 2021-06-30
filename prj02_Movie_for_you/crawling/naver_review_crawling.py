# crawling 작업

# crwaling은 각자 진행하고 빨리 완성되는 코드로 연도를 나눠서 진행함.
# 일단 2019년 개봉작만 크롤링하고 나머지는 연도별로 크롤링해서 합침.
# 데이터는 데이터프레임으로 작업하고 저장 형식은 csv로 함.
# 컬럼명은 ['titles', 'reviews']로 통일함.
# 파일명은 reviews_0000.csv로 하고, 0000은 연도로 함.
# 크롤링 파일은 https://url.kr/74bjw5에 업로드 함.

# 모듈 임포트!
from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time

# 구글 드라이브 옵션 설정
options = webdriver.ChromeOptions()
# options.add_argument('headless')
options.add_argument('disable_gpu')
options.add_argument('lang=ko_KR')

# 드라이버 불러오기 및 설정
driver = webdriver.Chrome('chromedriver', options=options)

# 영화제목과 리뷰를 담을 빈 리스트 생성
titles = []
reviews = []

# 작동하는지 테스트
# url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open=2019&page=43'
# driver.get(url)
# y = driver.find_elements_by_xpath('//*[@id="old_content"]/ul/li')
# print(len(y))
#
# driver.find_element_by_xpath(f'//*[@id="old_content"]/ul/li[1]/a').click()
# # driver.find_element_by_xpath(f'//*[@id="old_content"]/ul/li[%d]/a'%j) # 변수 지정 다른 방법
# time.sleep(0.5) # 페이지 로드를 위한 시간
# driver.find_element_by_xpath('//*[@id="movieEndTabMenu"]/li[6]/a/em').click()

# 페이지와 제목 크롤링
try:
    for i in range(1, 44): # 연도별 영화 리스트 페이지수 확인
        url = f'https://movie.naver.com/movie/sdb/browsing/bmovie.nhn?open=2019&page={i}'


        for j in range(1, 21): # 영화 제목 한 페이지당 20개
            try: # 제목
                driver.get(url)
                time.sleep(1)
                movie_title_xpath = f'//*[@id="old_content"]/ul/li[{j}]/a'
                title = driver.find_element_by_xpath(movie_title_xpath).text
                print(title)
                driver.find_element_by_xpath(movie_title_xpath).click()
                time.sleep(1)
                try: # 리뷰
                    btn_review_xpath = '//*[@id="movieEndTabMenu"]/li[6]/a/em'
                    driver.find_element_by_xpath(btn_review_xpath).click()  # 리뷰 버튼 클릭
                    time.sleep(1)
                    review_len_xpath = '//*[@id="reviewTab"]/div/div/div[2]/span/em' # 총 리뷰 수
                    review_len = driver.find_element_by_xpath(review_len_xpath).text

                    review_len = int(review_len) # 리뷰 길이 int 값으로
                    try:
                        for k in range(1, ((review_len-1) // 10)+2): # 리뷰 갯수로 최대 페이지 계산
                            review_page_xpath = f'//*[@id="pagerTagAnchor{k}"]/span' # 리뷰 페이지 버튼 xpath
                            driver.find_element_by_xpath(review_page_xpath).click() # 리뷰 페이지 클릭
                            time.sleep(1)
                            for l in range(1, 11): # 한 페이지에 있는 리뷰 모두 크롤링
                                review_title_xpath = f'//*[@id="reviewTab"]/div/div/ul/li[{l}]' # 리뷰 제목 xpath
                                try:
                                    driver.find_element_by_xpath(review_title_xpath).click() # 리뷰 제목 xpath 클릭
                                    time.sleep(1)
                                    try:
                                        review_xpath = '//*[@id="content"]/div[1]/div[4]/div[1]/div[4]' # 리뷰 내용 xpath
                                        review = driver.find_element_by_xpath(review_xpath).text
                                        titles.append(title)
                                        reviews.append(review)
                                        driver.back()
                                        time.sleep(1)
                                    except:
                                        driver.back()
                                        time.sleep(1)
                                        print('review crawling error')
                                except:
                                    time.sleep(1)
                                    print('review title click error')
                    except:
                        print('review page btn click error')
                except:
                    print('review btn click error')
            except NoSuchElementException:
                driver.get(url)
                time.sleep(1)
                print('NoSuchElementException')

        print(len(titles))
        df_review = pd.DataFrame({'titles': titles, 'reviews': reviews})
        df_review['years'] = 2019
        print(df_review.head(20))
        df_review.to_csv(f'./reviews_2019_{i}_page.csv')

except:
    print('except1')
finally:
    driver.close()