import os
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import time

# 配置参数：定义要爬取的动物种类和每种动物需要下载的图片数量
# 命名规范，“-”前面是大类，后面是常见的品种名。
ANIMALS = [
'猫类-狸花猫',  
'猫类-无毛猫',  
'猫类-橘猫',  
'猫类-布偶猫',   
'猫类-波斯猫',  
'猫类-缅因猫',  
'犬类-哈士奇',  
'犬类-藏獒',  
'犬类-贵宾犬',  
'犬类-金毛',  
'犬类-拉布拉多',  
'犬类-柯基',  
'犬类-柴犬',  
'犬类-牧羊犬',  
'犬类-吉娃娃',  
'鸟类-鹦鹉',  
'鸟类-金丝雀',  
'鸟类-鸽子',   
'鱼类-金鱼',  
'鱼类-锦鲤',   
'鱼类-小丑鱼',   
'小型哺乳动物-兔',   
'小型哺乳动物-仓鼠',  
'小型哺乳动物-龙猫',  
'爬行动物-巴西龟',  
'爬行动物-陆龟',     
'两栖动物-树蛙',  
'两栖动物-东方蝾螈',  
]

NUM_IMAGES_PER_ANIMAL = 150

# 文件夹路径配置
BASE_DIR = 'animal'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'val')

def create_directories():
    """
    创建必要的目录结构。
    如果目录不存在，则创建它们。
    """
    for folder in [BASE_DIR, TRAIN_DIR, VALIDATION_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 对于每种动物，为训练集和验证集创建子目录
    for animal in ANIMALS:
        for path in [TRAIN_DIR, VALIDATION_DIR]:
            animal_folder = os.path.join(path, animal)
            if not os.path.exists(animal_folder):
                os.makedirs(animal_folder)

def download_images(animal, num_images):
    """
    爬取指定数量的动物图片URL。
    使用Google图片搜索进行爬取，并处理翻页。
    """
    image_urls = []
    page = 0
    while len(image_urls) < num_images:
        # url = f'http://www.google.com/search?q={animal}&tbm=isch&start={page * 20}'
        url = f'https://cn.bing.com/images/search?q={animal}&form=HDRSC2&first={page+1}'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')
        
        for img in images:
            if len(image_urls) >= num_images:
                break
            # 过滤掉 base64 数据 URI:  not img['src'].startswith('data:image')
            # 过滤 Invalid URL '/sa/simg/Flag_Feedback.png'
            if 'src' in img.attrs and (img['src'].startswith('http') or img['src'].startswith('https')):
                image_urls.append(img['src'])
                # print(f'抓取地址 {img['src']}')
            elif 'data-src' in img.attrs and (img['data-src'].startswith('http') or img['src'].startswith('https')):
                image_urls.append(img['data-src'])
                # print(f'抓取地址 {img['data-src']}')
        
        page += 1
        time.sleep(2)  # 避免过于频繁请求导致被封禁
    
    return image_urls[:num_images]

def save_images(image_urls, animal):
    """
    将获取到的图片URL列表分为训练集和验证集，并保存图片到本地。
    """
    train_urls, val_urls = train_test_split(image_urls, test_size=0.2, random_state=42)
    
    def _save(urls, base_path):
        """内部函数，用于保存图片"""
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f'正在下载 {animal}_{i}.jpg')
                    with open(os.path.join(base_path, animal, f'{animal}_{i}.jpg'), 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"跳过图片 {url}，状态码: {response.status_code}")
            except Exception as e:
                print(f"无法下载图片 {url}，错误: {e}")
    
    # 分别保存训练集和验证集图片
    _save(train_urls, TRAIN_DIR)
    _save(val_urls, VALIDATION_DIR)


def main():
    """主程序入口"""
    create_directories()
    for animal in ANIMALS:
        print('开始爬取【', animal, '】的图片')
        image_urls = download_images(animal, NUM_IMAGES_PER_ANIMAL)
        print('开始下载【', animal, '】的图片')
        save_images(image_urls, animal)
    print("图片爬取和分类完成！")

if __name__ == '__main__':
    main()