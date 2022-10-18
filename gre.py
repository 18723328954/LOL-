from pyecharts.charts import Bar,Pie,Page

Page.save_resize_html(
    source='大屏_临时.html',
    cfg_file='./src/chart_config.json',
    dest="final.html"
)