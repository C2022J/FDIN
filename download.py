import openxlab
openxlab.login(ak='qmdag6ak2lxwpgrlyz1p', sk='lgeevyla3g2b6vd9qnrjb61akrzk8pnamznwoqmx') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/Middlebury_2014') #数据集信息查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/Middlebury_2014', target_path='/data') # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/Middlebury_2014',source_path='/README.md', target_path='/data') #数据集文件下载