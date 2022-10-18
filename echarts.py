import random
from LOL.data import *
from pyecharts.charts import Bar, Pie, Page, Line, HeatMap
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

## 模型预测概率
model_name = ['SVM', 'DecisionTree', 'RandomForest', 'MLP', 'Transformer']
train = [98.04, 97.58, 97.88, 97.12, 96.49]
test = [96.18, 95.34, 96.45, 96.90, 97.54]

## 召唤师技能使用
x_data = ['闪现', '引燃', '惩戒', '屏障', '疾跑', '传送', '虚弱', '净化', '治疗']
top = [100766, 30894, 3154, 3043, 19949, 25398, 12214, 696, 9846]
jungle = [60225, 23486, 100248, 324, 18637, 933, 214, 93, 1800]
mid = [100211, 34891, 3241, 8013, 22013, 14681, 12041, 987, 9882]
adc = [101150, 7819, 321, 19583, 14001, 4513, 2378, 25401, 30794]
support = [78940, 32415, 2481, 877, 10493, 17897, 37854, 5597, 19406]
# x_data=x_data[::-1]     #按照时间顺序排列

## 相关性热力图
attr = ['蓝赢', '红赢', '一血', '一塔', '一先锋', '一大龙', '一小龙', '一河蟹',
        '蓝队推塔数', '蓝队推水晶数', '蓝队大龙数', '蓝队小龙数', '蓝队先锋数', '红队推塔数',
        '红队推水晶数', '红队大龙数', '红队小龙数', '红队先锋数']
corr = getCorr()
corr = [[i, j, round(corr[i][j], 2)] for i in range(len(corr)) for j in range(len(corr))]

## 野区资源统计图
table = HeatMap()
headers = ['推塔数', '推水晶数', '大龙数', '小龙数', '先锋数']
rows = ['Blue', 'Red']
values = countResource()
value = [[i, j, values[i][j]] for i in range(len(rows)) for j in range(len(headers))]

## 英雄登场次数
l1 = ['卡莎', '盲僧', '酒桶', '霞', '泰坦', '瑞兹', '亚索', '船长', '诺', '天使']
hero_times = [13002, 12983, 13002, 10658, 9853, 9188, 8838, 8691, 7872, 8300]
l1.reverse()
hero_times.reverse()
## 获胜数量
bar = Bar(
    init_opts=opts.InitOpts(bg_color='rgba(6,48,109,.2)', width="450px", height="350px", chart_id='bar_cmt2'))  # 初始化条形图
bar.add_xaxis(['Blue', 'Red'], )  # 增加x轴数据
bar.add_yaxis("获胜次数", [26077, 25413])  # 增加y轴数据
bar.set_global_opts(
    legend_opts=opts.LegendOpts(pos_left='right'),
    title_opts=opts.TitleOpts(title="蓝红队伍获胜数", pos_left='left'),  # 标题
    toolbox_opts=opts.ToolboxOpts(is_show=False, ),  # 不显示工具箱
    xaxis_opts=opts.AxisOpts(name="队伍",  # x轴名称
                             axislabel_opts=opts.LabelOpts(font_size=16, color='white')),  # 字体大小
    yaxis_opts=opts.AxisOpts(name="获胜次数",
                             axislabel_opts=opts.LabelOpts(font_size=8, color='white'),
                             splitline_opts=opts.SplitLineOpts(is_show=True,
                                                               linestyle_opts=opts.LineStyleOpts(type_='solid')),
                             ),  # y轴名称
)
# 标记最大值
bar.set_series_opts(
    markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max", name="最大值"), ],
                                      symbol_size=35)  # 标记符号大小
)

## 比赛时长
pie = (
    Pie(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                chart_id='pie1'))
    .add(series_name="LOL比赛时长分布",  # 系列名称
         data_pair=countGameDuration(),
         rosetype="radius",  # 是否展示成南丁格尔图
         radius=["30%", "55%"],  # 扇区圆心角展现数据的百分比，半径展现数据的大小
         )  # 加入数据
    .set_global_opts(  # 全局设置项
        title_opts=opts.TitleOpts(title="LOL比赛时长", pos_left='left'),  # 标题
        legend_opts=opts.LegendOpts(pos_left='right', orient='vertical')  # 图例设置项,靠右,竖向排列
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")))  # 样式设置项

## 召唤师技能
line = Line(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                    chart_id='line1')).add_xaxis(xaxis_data=x_data)  # 添加x轴
line.add_yaxis(  # 第一条曲线
    series_name='top',
    y_axis=top,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
line.add_yaxis(  # 添加第二条曲线
    series_name='jungle',
    y_axis=jungle,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
line.add_yaxis(  # 添加第三条曲线
    series_name='mid',
    y_axis=mid,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
line.add_yaxis(  # 添加第三条曲线
    series_name='adc',
    y_axis=adc,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
line.add_yaxis(  # 添加第三条曲线
    series_name='support',
    y_axis=support,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)

line.set_global_opts(
    title_opts=opts.TitleOpts(title='召唤师技能携带数目', pos_top='top', pos_left='left'),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    yaxis_opts=opts.AxisOpts(
        name='次数',
        type_="value",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
    legend_opts=opts.LegendOpts(pos_top='40'),
    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False), )

## 相关性热力图
heatmap = HeatMap(
    init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="850px", bg_color='rgba(6,48,109,.2)', height="500px",
                            chart_id='heatmap1'))
heatmap.add_xaxis(attr)
heatmap.add_yaxis("", attr, corr, label_opts=opts.LabelOpts(is_show=True, position="inside"))
heatmap.set_global_opts(title_opts=opts.TitleOpts(title="数据之间相关性", pos_top='top', pos_left='center'),
                        visualmap_opts=opts.VisualMapOpts(max_=1, min_=-1, pos_right='right'),  # 视觉映射配置
                        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)))  # 将x轴标签旋转

## 野区资源统计图
table = HeatMap(
    init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="850px", bg_color='rgba(6,48,109,.2)', height="500px",
                            chart_id='table1'))
table.add_xaxis(rows)
table.add_yaxis("", headers, value, label_opts=opts.LabelOpts(is_show=True, position="inside", color='orange'))
table.set_global_opts(title_opts=opts.TitleOpts(title="各队资源数量", pos_top='top', pos_left='center'),
                      xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),
                      visualmap_opts=opts.VisualMapOpts(is_show=False, range_color=['rgba(6,48,109,.2)']))  # 将x轴标签旋转

## 英雄登场次数
re_bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="500px", bg_color='rgba(6,48,109,.2)', height="400px",
                                chart_id='re_bar'))
    .add_xaxis(l1)
    .add_yaxis('登场次数', hero_times)
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position="right"))
    .set_global_opts(title_opts=opts.TitleOpts(title="英雄登场次数"))
)

## 模型预测
bar1 = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="500px", bg_color='rgba(6,48,109,.2)', height="400px",
                                chart_id='bar1'))
    .add_xaxis(model_name)
    .add_yaxis("train", train)
    .add_yaxis("test", test)
    .set_global_opts(title_opts=opts.TitleOpts(title="模型预测比赛胜率"),
                     toolbox_opts=opts.BrushOpts(),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)))
)

line1 = Line(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                     chart_id='line1')).add_xaxis(xaxis_data=model_name)  # 添加x轴
line1.add_yaxis(  # 第一条曲线
    series_name='train',
    y_axis=train,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
line1.add_yaxis(  # 添加第二条曲线
    series_name='test',
    y_axis=test,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)

line1.set_global_opts(
    title_opts=opts.TitleOpts(title='模型预测比赛胜率', pos_top='top', pos_left='center'),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    yaxis_opts=opts.AxisOpts(
        name='概率',
        type_="value",
        axistick_opts=opts.AxisTickOpts(is_show=True),
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
    legend_opts=opts.LegendOpts(pos_top='40'),
    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=30)),
)

all = bar1.overlap(line1)

# 绘制:整个页面
page = Page(
    page_title="基于Python的英雄联盟数据分析",
    layout=Page.DraggablePageLayout,  # 拖拽方式
)
page.add(
    bar,
    line,
    pie,
    heatmap,
    table,
    re_bar,
    all
)
page.render('大屏_临时.html')  # 执行完毕后,打开临时html并排版,排版完点击Save Config，把json文件放到本目录下
print('生成完毕:大屏_临时.html')
